import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

class PhantomInkEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    MAX_TURNS = 8
    MAX_CLUES = 8
    MAX_CHARS = 12
    MAX_QUESTIONS = 5

    PHASE_MAP = {"DECISION": 0, "THINKING": 1, "WRITING": 2}

    def __init__(self, word_data, render_mode=None):
        super().__init__()
        self.word_data = word_data
        self.target_options = list(self.word_data.keys())
        # Assumes uniform question keys across words
        first_word = self.target_options[0]
        self.questions = list(self.word_data[first_word].keys())

        # NLP Setup - Moving to CPU/GPU explicitly for NumPy 2.0 compatibility with Torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        self.target_embeddings = self.model.encode(self.target_options, convert_to_tensor=True)
        self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)

        self.all_clue_words = sorted(list(set(
            val.upper() for data in word_data.values() for val in data.values()
        )))
        self.clue_embeddings = self.model.encode(self.all_clue_words, convert_to_tensor=True)

        # 0-4: Ask Question, 5: Request Letter, 6: End Clue, 7: Submit Guess
        self.action_space = spaces.Discrete(8)
        emb_dim = self.model.get_sentence_embedding_dimension()

        self.observation_space = spaces.Dict({
            "clues": spaces.Box(low=0, high=26, shape=(self.MAX_CLUES, self.MAX_CHARS), dtype=np.int32),
            "phase": spaces.Discrete(3),
            "turn": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "questions": spaces.Box(low=-np.inf, high=np.inf, shape=(self.MAX_QUESTIONS, emb_dim), dtype=np.float32)
        })

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # Gymnasium handles seeding via this call
        super().reset(seed=seed)

        # Use self.np_random instead of np.random for reproducibility
        self.target_word = self.np_random.choice(self.target_options).upper()
        self.clue_history = []
        self.current_turn = 0
        self.phase = "DECISION"

        self.spirit_answer = ""
        self.revealed_chars = ""
        self.predicted_word = ""
        self.guess_progress = 0

        self.all_q_indices = np.arange(len(self.questions))
        self.np_random.shuffle(self.all_q_indices)

        self.active_q_indices = self.all_q_indices[-self.MAX_QUESTIONS:].copy()
        self.unused_ptr = len(self.all_q_indices) - self.MAX_QUESTIONS - 1

        self.question_history = []
        self.active_turn_question = None

        self.lettersKnown = []

        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self.current_turn == self.MAX_TURNS - 1 and self.phase == "DECISION":
          action = 7

        if not self._is_action_valid(action) and self.phase != "WRITING":
            return self._get_obs(), -7.0, False, True, {"error": "invalid_action_phase"}

        if self.current_turn > self.MAX_TURNS:
            return self._get_obs(), -15.0, False, True, {"error": "out_of_turns"}

        if self.phase == "DECISION":
            if 0 <= action <= 4:
                actual_q_idx = self.active_q_indices[action]
                self.active_turn_question = self.question_embeddings[actual_q_idx]
                q_text = self.questions[actual_q_idx]

                self.spirit_answer = self.word_data[self.target_word][q_text].upper()

                # Update question hand
                new_q_idx = self.all_q_indices[self.unused_ptr]
                self.active_q_indices[action] = new_q_idx

                self.unused_ptr -= 1
                if self.unused_ptr < 0:
                    self.unused_ptr = len(self.all_q_indices) - self.MAX_QUESTIONS - 1

                self.phase = "THINKING"
                self.current_turn += 1
                reward = -1 * float(self.current_turn) / self.MAX_TURNS
                reward += 0.1

            elif action == 7:
                self.current_turn += 1
                self.phase = "WRITING"
                reward = 0.5

        elif self.phase == "THINKING":
            if action == 5:  # "Next Letter"
                if len(self.revealed_chars) < len(self.spirit_answer):
                    self.revealed_chars += self.spirit_answer[len(self.revealed_chars)]
                    reward = -0.05
                else:
                    self.phase = "DECISION"
                    reward = -0.5

            elif action == 6: # "Stop Writing"
                if self.revealed_chars:
                    self.clue_history.append(self.revealed_chars)
                    reward = 1.0 + (0.2 * len(self.revealed_chars))
                    self.question_history.append(self.active_turn_question)

                self.revealed_chars = ""
                self.active_turn_question = None
                self.phase = "DECISION"
            if self.current_turn == self.MAX_TURNS:
                reward = -15.0
                truncated = True

        elif self.phase == "WRITING":
            char = self._get_predicted_char()

            if char == self.target_word[self.guess_progress]:
                self.guess_progress += 1
                if len(self.lettersKnown) < self.guess_progress:
                  self.lettersKnown.append(char)
                reward = 5.0

                if self.guess_progress == len(self.target_word):
                    reward = 50.0
                    terminated = True
            else:
                reward = -0.5 * (self.guess_progress + 1)
                self.guess_progress = 0
                self.predicted_word = ""
                self.phase = "DECISION"
            if self.current_turn == self.MAX_TURNS:
                truncated = not terminated

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        grid = np.zeros((self.MAX_CLUES, self.MAX_CHARS), dtype=np.int32)

        for i, clue in enumerate(self.clue_history[:self.MAX_CLUES-1]):
            for j, char in enumerate(clue[:self.MAX_CHARS]):
                grid[i, j] = ord(char) - ord('A') + 1

        for j, char in enumerate(self.revealed_chars[:self.MAX_CHARS]):
            grid[self.MAX_CLUES-1, j] = ord(char) - ord('A') + 1

        active_embs = self.question_embeddings[self.active_q_indices].cpu().numpy()

        return {
            "clues": grid,
            "phase": self.PHASE_MAP[self.phase],
            "turn": np.array([float(self.current_turn) / self.MAX_TURNS], dtype=np.float32),
            "questions": active_embs.astype(np.float32)
        }

    def _is_action_valid(self, action):
        if self.phase == "DECISION":
            return (0 <= action <= 4) or action == 7
        if self.phase == "THINKING":
            return action in [5, 6]
        if self.phase == "WRITING":
            return action == 7
        return False

    def _get_predicted_char(self):
        # 1. Use existing predicted_word if it matches our current progress
        if self.predicted_word and self.guess_progress < len(self.predicted_word):
            # Verify the predicted word still matches what we know to be true
            known_prefix = "".join(self.lettersKnown)
            if self.predicted_word.startswith(known_prefix):
                return self.predicted_word[self.guess_progress]

        # 2. If no clues yet, we are just throwing darts
        if not self.clue_history:
            return chr(self.np_random.integers(65, 91))

        # 3. Filter the target options based on lettersKnown
        known_prefix = "".join(self.lettersKnown)
        valid_indices = [
            i for i, word in enumerate(self.target_options)
            if word.upper().startswith(known_prefix)
        ]

        # If for some reason no words match, fall back to all options
        # (though this shouldn't happen if lettersKnown is accurate)
        if not valid_indices:
            valid_indices = list(range(len(self.target_options)))

        # 4. Generate the context vector from clue history (your existing logic)
        combined_turn_vecs = []
        for fragment, q_emb in zip(self.clue_history, self.question_history):
            matches = [i for i, w in enumerate(self.all_clue_words) if w.startswith(fragment)]
            if matches:
                clue_vec = torch.mean(self.clue_embeddings[matches], dim=0)
                combined_vec = (clue_vec + 0.5 * q_emb) / 2
                combined_turn_vecs.append(combined_vec)

        if not combined_turn_vecs:
            return chr(self.np_random.integers(65, 91))

        context = torch.mean(torch.stack(combined_turn_vecs), dim=0).unsqueeze(0)

        # 5. Only score against words that match our known prefix
        filtered_embeddings = self.target_embeddings[valid_indices]
        scores = util.cos_sim(context, filtered_embeddings)[0]

        # Map the best score back to the original target_options list
        best_filtered_idx = torch.argmax(scores).item()
        actual_idx = valid_indices[best_filtered_idx]

        self.predicted_word = self.target_options[actual_idx].upper()

        # Return the next character in the best matching word
        if self.guess_progress < len(self.predicted_word):
            return self.predicted_word[self.guess_progress]
        return chr(self.np_random.integers(65, 91))