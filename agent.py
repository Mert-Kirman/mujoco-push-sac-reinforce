import torch
from torch import optim
import torch.nn.functional as F

from model import VPG


class REINFORCE_Agent():
    def __init__(self, lr=1e-3, gamma=0.99):
        self.model = VPG()
        self.learning_rate = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.gamma = gamma

        # Buffers to store trajectory data for the episode
        self.saved_log_probs = []
        self.rewards = []
        
    def decide_action(self, state):
        state = torch.tensor(state).to(torch.float32).unsqueeze(0)  # Add batch dimension

        # Forward pass through the network
        action_mean, act_std = self.model(state).chunk(2, dim=-1)

        # Softplus ensures the standard deviation is strictly positive
        action_std = F.softplus(act_std) + 1e-3  # increase variance to stimulate exploration

        action_dist = torch.distributions.Normal(action_mean, action_std)
        action = action_dist.sample()

        self.saved_log_probs.append(action_dist.log_prob(action).sum(dim=-1))
        return action.detach().squeeze(0).numpy()  # Remove batch dimension and convert to numpy
    
    def add_reward(self, reward):
        self.rewards.append(reward)

    def update_model(self):
        # Calculate discounted returns
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        # Baseline normalization
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate policy loss
        policy_loss = []
        for log_prob, R_t in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R_t)

        # Optimization step
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # Clear the buffers for the next episode
        del self.saved_log_probs[:]
        del self.rewards[:]
