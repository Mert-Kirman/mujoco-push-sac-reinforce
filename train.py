import os
import sys
from datetime import datetime
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from policy_grad_env import Policy_Grad_Env
from agent import REINFORCE_Agent as Agent
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent for pushing an object to a goal position in Mujoco.")
    parser.add_argument("--model", type=str, default="reinforce", choices=["reinforce", "sac"], help="Which model architecture to use for the agent.")
    parser.add_argument("--num_episodes", type=int, default=5000, help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush() # Force write to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{args.model}", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    model_path = os.path.join(run_dir, "model.pt")

    env = Policy_Grad_Env(render_mode="blind")
    
    agent = Agent(lr=3e-4, gamma=0.99)
    num_episodes = 5000 

    episode_rewards = []
    episode_rps = [] # Reward Per Step
    episode_final_dists = [] # Track the final Object-to-Goal distance
    best_avg_dist = float('inf') # We want to minimize the distance

    sys.stdout = Logger(os.path.join(run_dir, 'train_log.txt'))
    for episode in tqdm(range(args.num_episodes), desc=f"Training Agent with {args.model}"):
        state, _ = env.reset()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0

        while not done:
            # Action selection
            action = agent.decide_action(state)
            
            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Track rewards for the trajectory
            agent.add_reward(reward)
            cumulative_reward += reward
            
            # Check if episode has ended
            done = terminated or truncated
            
            state = next_state
            episode_steps += 1

        episode_rewards.append(cumulative_reward)
        episode_rps.append(cumulative_reward / max(episode_steps, 1))

        # Final object-to-goal distance in raw meters
        final_dist = env.raw_object_goal_distance()
        episode_final_dists.append(final_dist)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rps = np.mean(episode_rps[-100:])
            avg_dist = np.mean(episode_final_dists[-100:])
            tqdm.write(f"Episode {episode+1} | Avg Reward (last 100): {avg_reward:.2f} | Avg RPS: {avg_rps:.2f} | Avg Final Distance (last 100): {avg_dist:.4f}")

            # Save the model if it achieves a new low score in final object-to-goal distance
            if avg_dist < best_avg_dist:
                best_avg_dist = avg_dist
                torch.save(agent.model.state_dict(), model_path)
                tqdm.write(f"*** New Best {args.model} Model Saved (Avg Final Distance: {best_avg_dist:.4f}) ***")
        
        # Policy gradient update occurs strictly at the END of the episode
        agent.update_model()

    # Save training metrics as numpy arrays
    rewards_path = os.path.join(run_dir, "rewards.npy")
    rps_path = os.path.join(run_dir, "rps.npy")
    dists_path = os.path.join(run_dir, "final_dists.npy")
    np.save(rewards_path, np.array(episode_rewards))
    np.save(rps_path, np.array(episode_rps))
    np.save(dists_path, np.array(episode_final_dists))
    print(f"Training metrics saved")

    # Save hyperparameters and results
    hyperparams_path = os.path.join(run_dir, "config.txt")
    final_avg_reward = np.mean(episode_rewards[-100:])
    final_avg_rps = np.mean(episode_rps[-100:])
    final_avg_dist = np.mean(episode_final_dists[-100:])
    with open(hyperparams_path, 'w') as f:
        f.write(f"Training Run ({args.model}): {timestamp}\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  model_type: {args.model}\n")
        f.write(f"  gamma: {agent.gamma}\n")
        f.write(f"  learning_rate: {agent.learning_rate} (Adam, Constant)\n")
        f.write(f"  num_episodes: {args.num_episodes}\n")
        f.write(f"\nResults:\n")
        f.write(f"  final_avg_reward (last 100): {final_avg_reward:.2f}\n")
        f.write(f"  final_avg_rps (last 100): {final_avg_rps:.2f}\n")
        f.write(f"  max_reward: {max(episode_rewards):.2f}\n")
        f.write(f"  min_reward: {min(episode_rewards):.2f}\n")
        f.write(f"  final_avg_dist (last 100): {final_avg_dist:.4f}\n")
    print(f"Hyperparameters saved to {hyperparams_path}")
    
    # Plot Results
    figure_path = os.path.join(run_dir, "training_plot.png")
    plt.figure(figsize=(8, 12))
    
    # Plot 1: Cumulative Reward
    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards, alpha=0.6)
    window = min(100, len(episode_rewards))
    moving_avg_reward = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(episode_rewards)), moving_avg_reward, label=f'{window}-Episode Moving Avg')
    plt.title(f'{args.model} Cumulative Reward over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # Plot 2: Reward Per Step (RPS)
    plt.subplot(3, 1, 2)
    plt.plot(episode_rps, alpha=0.6)
    moving_avg_rps = np.convolve(episode_rps, np.ones(100)/100, mode='valid')
    plt.plot(moving_avg_rps, label='100-Episode Moving Avg')
    plt.title(f'{args.model} Reward Per Step (RPS) over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('RPS')
    plt.legend()
    
    # Plot 3: Final Object-to-Goal Distance (Lower is Better)
    plt.subplot(3, 1, 3)
    plt.plot(episode_final_dists, alpha=0.3, color='red')
    moving_avg_dist = np.convolve(episode_final_dists, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(episode_final_dists)), moving_avg_dist, color='darkred', label=f'{window}-Episode Moving Avg')
    plt.title(f'{args.model} Final Object-to-Goal Distance (Lower is Better)')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(figure_path)
    print(f"Training complete! Plot saved to {figure_path}")
