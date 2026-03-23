import torch
from stable_baselines3 import PPO
import os
from env import PhantomInkEnv
from wordData import word_data


save_path = "./logs/best_model/"
# 1. Load the Best Model
model_path = os.path.join(save_path, "best_model.zip")
model = PPO.load(model_path)

# 2. Create a single-process environment for testing
# We use the raw class so we can access internal variables for better printing
eval_env = PhantomInkEnv(word_data=word_data)

num_episodes = 5
action_names = {
    0: "Ask Q1", 1: "Ask Q2", 2: "Ask Q3", 3: "Ask Q4", 4: "Ask Q5",
    5: "Next Letter", 6: "Stop Writing", 7: "Guess"
}

for ep in range(num_episodes):
    obs, info = eval_env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print(f"\n--- Episode {ep + 1} | Target: {eval_env.target_word} ---")

    while not done:
        # Use deterministic=True to see the "optimal" strategy
        action, _states = model.predict(obs, deterministic=True)

        # Take the step
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        # Print the "Story" of the turn
        phase_name = ["DECISION", "THINKING", "WRITING"][obs["phase"]]
        print(f"Step {step_count} | Turn: {obs["turn"][0]} | Action: {action_names[int(action)]:12} | Phase: {phase_name:8} | Reward: {reward:+.2f}")

        if terminated:
            print(f"WIN! Word guessed in {eval_env.current_turn} turns!")
        elif truncated:
            print(f"GAME OVER: Reached turn/step limit.")

    print(f"Total Episode Reward: {total_reward:.2f}")


# Important to note that when Agent is forced to enter writing phase
# It might be making random actions, but those are ignored
# since the model is forced to go through the writing word phase regardless of the actions it chooses