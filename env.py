import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

class PhantomInkEnv(gym.Env):
    """
    A Gymnasium environment for the game Phantom Ink.
    The agent acts as a medium trying to guess a target word by asking questions 
    and receiving partial clues.
    """
    
    # --- Constants ---
    MAX_TURNS = 8.0
    MAX_CLUES = 8
    MAX_CHARS = 12
    MAX_QUESTIONS = 5
    PHASE_MAP = {"DECISION": 0, "THINKING": 1, "WRITING": 2}

    def __init__(self, word_data, render_mode=None):
        super().__init__()
        
        # 1. Data Setup
        self.word_data = word_data
        self.target_options = list(self.word_data.keys())
        self.questions = list(self.word_data[self.target_options[0]].keys())

        # 2. NLP Model Setup (CPU for compatibility)
        self.device = "cpu" 
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        # 3. Embedding Cache (Pre-calculation for speed)
        self._precompute_embeddings()

        # 4. Gymnasium Spaces
        self.action_space = spaces.Discrete(8)
        emb_dim = self.model.get_sentence_embedding_dimension()
        self.observation_space = spaces.Dict({
            "clues":         spaces.Box(low=0, high=26, shape=(8, 12), dtype=np.int32),
            "phase":         spaces.Discrete(3),
            "turn":          spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "questions":     spaces.Box(low=-np.inf, high=np.inf, shape=(5, emb_dim), dtype=np.float32),
            "question_mask": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int32)
        })

    def _precompute_embeddings(self):
        """Encodes vocabulary and expected answers into tensors once at startup."""
        # Vocabulary
        self.all_clue_words = sorted(list(set(
            val.upper() for data in self.word_data.values() for val in data.values()
        )))
        self.clue_vocab_embeddings = self.model.encode(self.all_clue_words, convert_to_tensor=True)

        # Target Clues
        dim = self.model.get_sentence_embedding_dimension()
        self.expected_embeddings = torch.zeros(
            (len(self.target_options), len(self.questions), dim), device=self.device
        )
        for t_idx, target in enumerate(self.target_options):
            clues = [self.word_data[target][q].upper() for q in self.questions]
            self.expected_embeddings[t_idx] = self.model.encode(clues, convert_to_tensor=True)

        # Questions
        self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)

    def reset(self, seed=None, options=None):
        """Resets the environment state for a new episode."""
        super().reset(seed=seed)
        
        # Target Selection
        self.target_word = self.np_random.choice(self.target_options).upper()
        
        # Game State
        self.current_turn = 0
        self.phase = "DECISION"
        self.clue_history = []
        self.q_text_history = []
        self.revealed_chars = ""
        self.guess_progress = 0
        self.active_q_text = None
        self.spirit_answer = ""
        self.prev_turn_guessed = False

        # Question Deck
        self.all_q_indices = np.arange(len(self.questions))
        self.np_random.shuffle(self.all_q_indices)
        self.active_q_indices = self.all_q_indices[:self.MAX_QUESTIONS].copy()
        
        return self._get_obs(), {}

    def step(self, action):
        """Applies the action and transitions the environment state."""
        reward = 0.0
        terminated = False
        truncated = False

        # 1. Edge Case: Forced Guess
        if self.current_turn == self.MAX_TURNS - 1 and self.phase == "DECISION":
            action = 7

        # 2. Validation Checks
        if not self._is_action_valid(action) and self.phase != "WRITING":
            return self._get_obs(), -50.0, False, True, {"error": "invalid_action_phase"}

        if self.current_turn > self.MAX_TURNS:
            return self._get_obs(), -50.0, False, True, {"error": "out_of_turns"}

        if self.phase == "DECISION" and action == 7 and len(self.clue_history) == 0:
            return self._get_obs(), -50.0, False, True, {"error": "no_info_guess"}

        # 3. Main State Machine
        if self.phase == "DECISION":
            reward = self._handle_decision_phase(action)
        elif self.phase == "THINKING":
            reward, truncated = self._handle_thinking_phase(action)
        elif self.phase == "WRITING":
            reward, terminated = self._handle_writing_phase(action)

        return self._get_obs(), float(reward), terminated, truncated, {}

    # --- Phase Handlers ---

    def _handle_decision_phase(self, action):
        if 0 <= action <= 4:
            q_idx = self.active_q_indices[action]
            self.active_q_text = self.questions[q_idx]
            self.spirit_answer = self.word_data[self.target_word][self.active_q_text].upper()
            self.phase = "THINKING"
            self.current_turn += 1
            self.prev_turn_guessed = False
            return 0.5
        elif action == 7:
            self.prev_turn_guessed = True
            self.current_turn += 1
            self.phase = "WRITING"
            return 0.0
        return 0.0

    def _handle_thinking_phase(self, action):
        reward = 0.0
        truncated = False
        
        if action == 5: # Next Letter
            if len(self.revealed_chars) < len(self.spirit_answer):
                self.revealed_chars += self.spirit_answer[len(self.revealed_chars)]
                reward = -0.1
            else:
                self.phase = "DECISION"
        elif action == 6: # Stop Writing
            if self.revealed_chars:
                self.clue_history.append(self.revealed_chars)
                self.q_text_history.append(self.active_q_text)
                reward = 1.0 + (0.1 * len(self.revealed_chars))
            self.revealed_chars = ""
            self.phase = "DECISION"
        
        if self.current_turn == self.MAX_TURNS:
            reward = -15.0
            truncated = True
        return reward, truncated

    def _handle_writing_phase(self, action):
        reward = 0.0
        terminated = False
        
        char = self._get_predicted_char()
        if char == self.target_word[self.guess_progress]:
            self.guess_progress += 1
            reward = 5.0
            if self.guess_progress == len(self.target_word):
                reward = 100.0
                terminated = True
        else:
            reward = -15.0
            self.guess_progress = 0
            self.phase = "DECISION"
            if self.current_turn == self.MAX_TURNS:
                terminated = True
        return reward, terminated

    # --- Helper Methods ---

    def _get_obs(self):
        """Constructs the observation dictionary for the agent."""
        grid = np.zeros((8, 12), dtype=np.int32)
        
        # Fill clue history into grid
        for i, clue in enumerate(self.clue_history[:7]):
            for j, char in enumerate(clue[:12]):
                grid[i, j] = ord(char) - ord('A') + 1
        
        # Current partial clue in the last row
        for j, char in enumerate(self.revealed_chars[:12]):
            grid[7, j] = ord(char) - ord('A') + 1

        active_embs = self.question_embeddings[self.active_q_indices].detach().cpu().numpy()

        return {
            "clues":         grid,
            "phase":         self.PHASE_MAP[self.phase],
            "turn":          np.array([self.current_turn / self.MAX_TURNS], dtype=np.float32),
            "questions":     active_embs.astype(np.float32),
            "question_mask": np.ones(5, dtype=np.int32)
        }

    def _is_action_valid(self, action):
        """Checks if the chosen action is legal in the current game state."""
        if self.phase == "DECISION":
            return (0 <= action <= 4) or (action == 7 and not self.prev_turn_guessed)
        if self.phase == "THINKING":
            return action in [5, 6]
        if self.phase == "WRITING":
            return action == 7
        return False

    def _get_predicted_char(self):
        """Simulates the medium's internal deduction using semantic similarity."""
        known_prefix = self.target_word[:self.guess_progress]
        valid_indices = [i for i, t in enumerate(self.target_options) if t.upper().startswith(known_prefix)]

        if self.clue_history:
            SIM_THRESHOLD = 0.6
            for fragment, q_text in zip(self.clue_history, self.q_text_history):
                q_idx = self.questions.index(q_text)
                match_vocab_mask = [w.startswith(fragment) for w in self.all_clue_words]
                if not any(match_vocab_mask): continue
                
                candidate_embs = self.clue_vocab_embeddings[match_vocab_mask]
                current_filtered = []
                for t_idx in valid_indices:
                    expected_emb = self.expected_embeddings[t_idx, q_idx].unsqueeze(0)
                    scores = util.cos_sim(expected_emb, candidate_embs)
                    if torch.max(scores) >= SIM_THRESHOLD:
                        current_filtered.append(t_idx)
                if current_filtered:
                    valid_indices = current_filtered

        chosen_idx = self.np_random.choice(valid_indices) if valid_indices else self.np_random.integers(len(self.target_options))
        predicted_word = self.target_options[chosen_idx].upper()
        
        return predicted_word[self.guess_progress] if self.guess_progress < len(predicted_word) else " "