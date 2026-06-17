import torch
import torchvision.transforms as transforms
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import environment
from agent import Agent

# Inherit from both BaseEnv and gym.Env
class Hw3Env(environment.BaseEnv, gym.Env):
    def __init__(self, **kwargs) -> None:
        environment.BaseEnv.__init__(self, **kwargs)
        self._delta = 0.05
        self._goal_thresh = 0.075 
        self._max_timesteps = 300 
        self._prev_obj_pos = None 

        # Gymnasium Space Definitions
        # Action Space: 2 continuous values (x, y movement) between -1.0 and 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)a
        
        # Observation Space: 6 continuous values (ee_pos, obj_pos, goal_pos)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene
    
    def reset(self, seed=None, options=None):
        # Gym environments expect reset to handle seeds. 
        # BaseEnv's reset doesn't take a seed naturally, so we wrap it carefully.
        environment.BaseEnv.reset(self)
        self._prev_obj_pos = self.data.body("obj1").xpos[:2].copy()
        self._t = 0

        # Gymnasium reset returns (observation, info_dict)
        try:
            return self.high_level_state(), {}
        except:
            return np.zeros(6, dtype=np.float32), {}

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]

        d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        # distance-based rewards
        r_ee_to_obj = -0.1 * d_ee_to_obj  # getting closer to object
        r_obj_to_goal = -0.2 * d_obj_to_goal  # moving object to goal

        # direction bonus
        obj_movement = obj_pos - self._prev_obj_pos
        dir_to_goal = (goal_pos - obj_pos) / (np.linalg.norm(goal_pos - obj_pos) + 1e-8)
        r_direction = 0.5 * max(0, np.dot(obj_movement / (np.linalg.norm(obj_movement) + 1e-8), dir_to_goal))
        if np.linalg.norm(obj_movement) < 1e-6:  # Avoid division by zero
            r_direction = 0.0

        # terminal bonus
        r_terminal = 10.0 if self.is_terminal() else 0.0

        r_step = -0.1  # penalty for each step

        self._prev_obj_pos = obj_pos.copy()
        return r_ee_to_obj + r_obj_to_goal + r_direction + r_terminal + r_step

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps
    
    def step(self, action):
        # Clip action to defined space just in case, then scale
        action = np.clip(action, self.action_space.low, self.action_space.high)
        scaled_action = action * self._delta
        
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + scaled_action, [0.25, -0.3], [0.75, 0.3])
        
        result = self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)            

        self._t += 1

        state = self.high_level_state()
        reward = self.reward()
        terminated = self.is_terminal()
        
        if result:  
            truncated = self.is_truncated()
        else:  
            truncated = True
            
        # Gymnasium step returns 5 values: obs, reward, terminated, truncated, info
        info = {}
        return state, reward, terminated, truncated, info

if __name__ == "__main__":
    env = Hw3Env(render_mode="offscreen")
    
    agent = Agent(lr=3e-4, gamma=0.99)
    num_episodes = 5000 

    rews = []

    for i in range(num_episodes):        
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

        print(f"Episode={i:04d} | Steps={episode_steps:03d} | Reward={cumulative_reward:.3f}")
        rews.append(cumulative_reward)
        
        # Policy gradient update occurs strictly at the END of the episode
        agent.update_model()

    torch.save(agent.model.state_dict(), "model.pt")
    np.save("rews.npy", np.array(rews))
