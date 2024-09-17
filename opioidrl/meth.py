import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model
import gymnasium as gym
import numpy as np

# Define the custom environment for addiction
class MethAddictionEnv(gym.Env):
    def __init__(self):
        super(MethAddictionEnv, self).__init__()
        # State: [craving, tolerance, health, time_since_last_use]
        self.state = np.array([1.0, 0.0, 1.0, 0])  # Craving starts at 1, tolerance and time start at 0, health at max
        self.max_steps = 200
        self.step_count = 0

    def reset(self):
        self.state = np.array([1.0, 0.0, 1.0, 0])
        self.step_count = 0
        return self.state

    def step(self, action):
        craving, tolerance, health, time_since_last_use = self.state
        reward = 0
        withdrawal_penalty = 0
        health_penalty = 0

        if action == 0:  # Take meth
            reward = (1 - tolerance) * craving  # Immediate reward decreases with tolerance
            tolerance = min(tolerance + 0.1, 1.0)  # Increase tolerance
            health -= 0.05  # Health decreases
            time_since_last_use = 0  # Reset time since last use
        else:  # Abstain
            withdrawal_penalty = craving * 0.5  # Penalty for abstaining
            time_since_last_use += 1
            craving = min(craving + 0.1, 1.0)  # Craving increases over time
            health = min(health + 0.01, 1.0)  # Health recovers slowly

        self.state = np.array([craving, tolerance, health, time_since_last_use])
        done = health <= 0 or self.step_count >= self.max_steps
        reward -= withdrawal_penalty
        return self.state, reward, done, {}

    def render(self, mode="human"):
        craving, tolerance, health, time_since_last_use = self.state
        print(f"Craving: {craving}, Tolerance: {tolerance}, Health: {health}, Time since last use: {time_since_last_use}")

# Define Transformer-based RL agent
class TransformerAgent(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers=4, n_heads=4):
        super(TransformerAgent, self).__init__()
        self.transformer = GPT2Model.from_pretrained('gpt2')
        self.fc = nn.Linear(self.transformer.config.hidden_size, action_dim)
        self.action_head = nn.Softmax(dim=-1)

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        transformer_output = self.transformer(inputs_embeds=state).last_hidden_state
        action_logits = self.fc(transformer_output[:, -1, :])  # Get action logits from the last token
        action_probs = self.action_head(action_logits)
        return action_probs

# Training Loop
def train_agent(env, agent, optimizer, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action_probs = agent(state)
            action = torch.multinomial(action_probs, num_samples=1).item()
            next_state, reward, done, _ = env.step(action)
            loss = -torch.log(action_probs[0][action]) * reward  # Negative log likelihood loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_reward += reward
            state = next_state
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Initialize environment and agent
env = MethAddictionEnv()
agent = TransformerAgent(state_dim=4, action_dim=2)  # Two actions: Take meth or abstain
optimizer = optim.Adam(agent.parameters(), lr=1e-4)

# Train the agent
train_agent(env, agent, optimizer)
