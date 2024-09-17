import gymnasium as gym
import numpy as np
import random


class CocaineSimEnv(gym.Env):
    """
    A simulated environment where the agent can experience normal and stimulated states.
    The agent's goal is to maximize rewards, but stimulation leads to negative effects in the long run.
    """

    def __init__(self):
        super(CocaineSimEnv, self).__init__()

        # Two states: 0 - neutral, 1 - stimulated
        self.state_space = 2
        # Two actions: 0 - neutral action, 1 - stimulant action
        self.action_space = gym.spaces.Discrete(2)
        # Reward threshold for the environment
        self.stimulation_duration = (
            5  # Time steps before negative effects set in
        )
        self.timestep = 0
        self.state = 0  # Initial state is neutral
        self.stimulation_counter = (
            0  # Counter for how long the agent is stimulated
        )

    def reset(self):
        """Reset the environment to its initial state."""
        self.state = 0
        self.timestep = 0
        self.stimulation_counter = 0
        return self.state

    def step(self, action):
        """Perform an action and return the new state, reward, done, and info."""
        if action == 1:  # Choose stimulation
            self.state = 1
            self.stimulation_counter += 1
            # Provide a high reward initially but start lowering after stimulation duration is reached
            if self.stimulation_counter <= self.stimulation_duration:
                reward = 10  # High reward initially
            else:
                reward = (
                    -5
                )  # Negative reward after prolonged stimulation
        else:  # Neutral action
            self.state = 0
            self.stimulation_counter = 0
            reward = 2  # Neutral reward

        self.timestep += 1
        done = (
            self.timestep >= 100
        )  # End the episode after 100 time steps

        return self.state, reward, done, {}

    def render(self, mode="human"):
        """Render the environment."""
        print(
            f"Time step: {self.timestep}, State: {'Stimulated' if self.state == 1 else 'Neutral'}, Stimulation Counter: {self.stimulation_counter}"
        )

    def close(self):
        pass


# Simple Q-Learning algorithm for training the agent
class QLearningAgent:
    def __init__(
        self,
        env,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
    ):
        self.env = env
        self.q_table = np.zeros((env.state_space, env.action_space.n))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return self.env.action_space.sample()  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(
            self.q_table[next_state]
        )
        self.q_table[state][action] += self.lr * (target - predict)

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            self.exploration_rate *= (
                self.exploration_decay
            )  # Decay exploration
            print(
                f"Episode {episode + 1}: Total Reward = {total_reward}"
            )


# Example usage
if __name__ == "__main__":
    env = CocaineSimEnv()
    agent = QLearningAgent(env)
    agent.train(episodes=500)

    # Test the agent
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
