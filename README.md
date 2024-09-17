[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# Opioid RL

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


**OpioidRL** is a cutting-edge reinforcement learning (RL) library that simulates drug addiction behaviors within RL agents. Inspired by the addictive properties of drugs like methamphetamine and crack cocaine, OpioidRL offers a unique environment where agents experience reward dependency, high-risk decision-making, and compulsive behaviors — pushing RL research into new and provocative territories.

## Features

- **Meth Simulation**: Models the erratic and compulsive high-risk behaviors typically seen in methamphetamine addiction.
- **Crack Simulation**: Models the short-term, intense craving for rewards, leading to aggressive reward-seeking behaviors.
- **Customizable Reward Loops**: Easily adjust the reinforcement pathways to mimic varying levels of addiction, from mild dependency to extreme compulsion.
- **Addiction Dynamics**: Introduces tolerance, withdrawal, and relapse phenomena, simulating real-world addiction cycles.
- **Compatible with Any RL Framework**: Easily integrate OpioidRL with popular RL frameworks like PyTorch, TensorFlow, and Stable Baselines3.

## Installation

You can install OpioidRL using `pip`:

```bash
pip install opioidrl
```

## Quick Start

Below is a simple example of how to integrate **OpioidRL** into your RL pipeline.

```python
import opioidrl
from stable_baselines3 import PPO

# Create a Crack environment
env = opioidrl.make('Crack-v0')

# Train the agent using PPO
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Test the agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

### Example: Meth Environment

```python
import opioidrl
from stable_baselines3 import A2C

# Create a Meth environment
env = opioidrl.make('Meth-v0')

# Train the agent using A2C
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate agent behavior
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

## Available Environments

OpioidRL currently offers two environments simulating different types of addiction:

1. **Crack-v0**: Fast and intense, simulates the short-term, high-risk reward-seeking behaviors common in crack cocaine addiction.
2. **Meth-v0**: More sustained compulsive behaviors, with agents showing an increasing tolerance and willingness to take extreme actions for delayed rewards.

### Environment Customization

You can modify the parameters of each environment to simulate different levels of addiction severity:

```python
env = opioidrl.make('Meth-v0', tolerance_increase_rate=0.01, withdrawal_penalty=5)
```

### Configuration Options

- `tolerance_increase_rate`: How fast the agent builds tolerance to rewards.
- `withdrawal_penalty`: The penalty imposed when the agent doesn't receive its expected reward.
- `relapse_probability`: The probability that an agent will fall back into compulsive behaviors after overcoming addiction.

## Roadmap

- **Opioid-v0**: A new environment simulating opioid addiction with prolonged reward dependency and extreme withdrawal effects.
- **Alcohol-v0**: An environment simulating long-term, mild addiction behaviors with subtle but persistent effects on decision-making.
- **Nicotine-v0**: Simulating the reward-seeking behavior tied to nicotine addiction, with frequent, small rewards.

## Contributing

Contributions are welcome! If you have ideas for new environments or features, feel free to submit a pull request or open an issue.

### Steps to Contribute:

1. Fork this repository.
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

OpioidRL is a research tool designed for educational and experimental purposes. The behaviors simulated within this library are based on abstract models of addiction and are not intended to trivialize or promote drug addiction in any form. Addiction is a serious issue, and if you or someone you know is struggling with addiction, please seek professional help.

Made with ❤️ by the OpioidRL team.
