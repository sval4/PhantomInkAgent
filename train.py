import os
import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from env import PhantomInkEnv
from wordData import word_data
from stable_baselines3.common.env_util import make_vec_env

save_path = "./logs/best_model/"
os.makedirs(save_path, exist_ok=True)


device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device_type.upper()}")

# env = make_vec_env(lambda: PhantomInkEnv(word_data=word_data), n_envs=3)
env = PhantomInkEnv(word_data=word_data)


eval_callback = EvalCallback(
    env, 
    best_model_save_path=save_path,
    log_path="./logs/", 
    eval_freq=5000,           
    deterministic=True, 
    render=False
)


model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,   
    n_steps=2048,         # How much data to collect before updating
    batch_size=64,        
    n_epochs=10,          
    ent_coef=0.1,         # Encourages exploration so it doesn't get stuck
    gae_lambda=0.95,
    clip_range=0.2,
    device=device_type,   
)

print("Starting training with Best Model tracking...")
model.learn(total_timesteps=500000, callback=eval_callback)