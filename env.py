import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

class PhantomInkEnv(gym.Env):
    MAX_TURNS = 10
    MAX_CLUES = 8
    MAX_CHARS = 12
    
    PHASE_MAP = {"DECISION": 0, "THINKING": 1, "WRITING": 2}

    def __init__(self, word_data):
        super().__init__()
        self.word_data = word_data
        self.target_options = list(word_data.keys())
        
        # NLP Setup
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.target_embeddings = self.model.encode(self.target_options, convert_to_tensor=True)

        # Build a flat list of all unique clue words in the dataset for the "Spirit" to use
        self.all_clue_words = sorted(list(set(
            val.upper() for data in word_data.values() for val in data.values()
        )))
        self.clue_embeddings = self.model.encode(self.all_clue_words, convert_to_tensor=True)

        # 0-4: Ask Question, 5: Request Letter, 6: End Clue, 7: Submit Guess
        self.action_space = spaces.Discrete(8)
        
        self.observation_space = spaces.Dict({
            "clues": spaces.Box(low=0, high=26, shape=(self.MAX_CLUES, self.MAX_CHARS), dtype=np.int32),
            "phase": spaces.Discrete(3),
            "turn": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32) 
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.target_word = np.random.choice(self.target_options).upper()
        self.clue_history = []
        self.current_turn = 0
        self.phase = "DECISION"
        
        # Spirit's state
        self.spirit_answer = ""
        self.revealed_chars = ""
        
        # Guesser's state
        self.predicted_word = ""
        self.guess_progress = 0 
        
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        
        # Helper to validate if the action matches the current game phase
        if not self._is_action_valid(action):
            return self._get_obs(), -10.0, True, False, {}

        if self.phase == "DECISION":
            if 0 <= action <= 4:
                # Handle Question Selection
                questions = list(self.word_data[self.target_word].keys())
                # If the dataset has fewer than 5 questions, wrap around or default
                q_text = questions[action % len(questions)]
                
                self.spirit_answer = self.word_data[self.target_word][q_text].upper()
                self.phase = "THINKING"
                self.current_turn += 1
                reward = -0.1 * self.current_turn
            
            elif action == 7:
                self.phase = "WRITING"

        elif self.phase == "THINKING":
            if action == 5:  # "Next Letter"
                if len(self.revealed_chars) < len(self.spirit_answer):
                    self.revealed_chars += self.spirit_answer[len(self.revealed_chars)]
                    reward = -0.05 
                else:
                    reward = -0.5 # Penalty for asking for letters that don't exist
            
            elif action == 6: # "Stop Writing"
                if self.revealed_chars:
                    self.clue_history.append(self.revealed_chars)
                    reward = 1.0 + (0.2 * len(self.revealed_chars))
                
                self.revealed_chars = ""
                self.phase = "DECISION"

        elif self.phase == "WRITING":
            # The agent is committing to letters of their predicted target word
            char = self._get_predicted_char()
            
            if char == self.target_word[self.guess_progress]:
                self.guess_progress += 1
                reward = 5.0
                
                if self.guess_progress == len(self.target_word):
                    reward = 50.0
                    terminated = True
            else:
                # Failed guess attempt
                reward = -5.0
                self.guess_progress = 0
                self.predicted_word = ""
                self.phase = "DECISION"

        if self.current_turn >= self.MAX_TURNS:
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # Create an 8x12 grid where letters are 1-26 (0 is empty)
        grid = np.zeros((self.MAX_CLUES, self.MAX_CHARS), dtype=np.int32)
        
        # Populate previous clues
        for i, clue in enumerate(self.clue_history[:self.MAX_CLUES-1]):
            for j, char in enumerate(clue[:self.MAX_CHARS]):
                grid[i, j] = ord(char) - ord('A') + 1
        
        # Populate currently active clue in the last row
        for j, char in enumerate(self.revealed_chars[:self.MAX_CHARS]):
            grid[self.MAX_CLUES-1, j] = ord(char) - ord('A') + 1

        return {
            "clues": grid,
            "phase": self.PHASE_MAP[self.phase],
            "turn": np.array([float(self.current_turn) / self.MAX_TURNS], dtype=np.float32)
        }

    def _is_action_valid(self, action):
        """Simple check to ensure agent isn't trying to 'Stop Writing' when not thinking, etc."""
        if self.phase == "DECISION":
            return (0 <= action <= 4) or action == 7
        if self.phase == "THINKING":
            return action in [5, 6]
        if self.phase == "WRITING":
            return action == 7
        return False

    def _get_predicted_char(self):
        """
        Uses semantic similarity to guess the target word based on clue fragments.
        If no clues exist, it picks a random letter.
        """
        if self.predicted_word and self.guess_progress < len(self.predicted_word):
            return self.predicted_word[self.guess_progress]

        if not self.clue_history:
            return chr(np.random.randint(65, 91))

        # Semantic Inference:
        # For every partial clue, find full words that start with those letters, 
        # average their embeddings, and compare against possible target words.
        clue_vecs = []
        for fragment in self.clue_history:
            matches = [i for i, w in enumerate(self.all_clue_words) if w.startswith(fragment)]
            if matches:
                avg_vec = torch.mean(self.clue_embeddings[matches], dim=0)
                clue_vecs.append(avg_vec)

        if not clue_vecs:
            return chr(np.random.randint(65, 91))

        context = torch.mean(torch.stack(clue_vecs), dim=0).unsqueeze(0)
        scores = util.cos_sim(context, self.target_embeddings)[0]
        
        self.predicted_word = self.target_options[torch.argmax(scores).item()]
        return self.predicted_word[self.guess_progress]