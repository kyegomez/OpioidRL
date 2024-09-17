import gymnasium as gym
import numpy as np
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from loguru import logger
from stable_baselines3 import PPO
from typing import Tuple


# Set up the Hugging Face GPT-2 model and tokenizer for text generation and reward shaping
class GPT2TextRewardModel:
    def __init__(self, model_name: str = "gpt2"):
        logger.info(
            "Loading GPT-2 model and tokenizer for text reward model..."
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        logger.info(f"Model {model_name} successfully loaded!")

    def generate_text(self, prompt: str) -> str:
        logger.debug(f"Generating text based on prompt: {prompt}")
        outputs = self.generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
        )
        generated_text = outputs[0]["generated_text"]
        logger.debug(f"Generated text: {generated_text}")
        return generated_text

    def reward(self, prompt: str) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        reward_value = (
            -loss.item()
        )  # Minimize the loss (negative reward)
        logger.debug(
            f"Reward calculated: {reward_value} for prompt: {prompt}"
        )
        return reward_value


# Modified reward function with creative effects
def creative_effects_reward(
    base_reward: float, text_length: int, intensity: float = 1.0
) -> float:
    """
    Modifies reward to simulate creativity and unpredictability.
    Introduces variance based on randomness and novelty.
    """
    variance = random.uniform(-0.5, 0.5) * intensity
    reward = base_reward + variance

    if text_length > 5 and random.random() > 0.7:
        spike = intensity * random.uniform(0.5, 2.0)
        reward += spike
        logger.debug(
            f"Reward spike applied: {spike} for text length: {text_length}"
        )

    if text_length < 3:  # Penalize short or "safe" choices
        penalty = 0.3 * intensity
        reward -= penalty
        logger.debug(
            f"Penalty applied for short text length: {penalty}"
        )

    logger.debug(f"Final reward with creative effects: {reward}")
    return reward


class GPT2TextGenerationEnv(gym.Env):
    def __init__(self, max_steps: int = 50):
        super(GPT2TextGenerationEnv, self).__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.llm = GPT2TextRewardModel()  # Hugging Face GPT-2 model
        self.action_space = gym.spaces.Discrete(
            50257
        )  # GPT-2 token space (vocab size)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.text = ""
        self.intensity = 1.0
        logger.info(
            f"GPT2TextGenerationEnv initialized with max_steps={max_steps}"
        )

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Convert action (token ID) back into a word/token
        word = self.llm.tokenizer.decode(
            [action], skip_special_tokens=True
        )
        self.text += word
        self.current_step += 1

        # Base reward from GPT-2 model (negative log-likelihood)
        base_reward = self.llm.reward(self.text)
        # Apply creative effects to the reward
        reward = creative_effects_reward(
            base_reward, len(self.text), intensity=self.intensity
        )

        done = self.current_step >= self.max_steps
        truncated = False

        logger.info(
            f"Step {self.current_step}/{self.max_steps}: action={action}, reward={reward}, done={done}"
        )
        logger.debug(f"Current generated text: {self.text}")

        return (
            np.array([reward], dtype=np.float32),
            reward,
            done,
            truncated,
            {},
        )

    def reset(
        self, seed: int = None, options: dict = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.text = ""
        self.intensity = random.uniform(1.0, 2.5)
        logger.info(
            f"Environment reset. New intensity: {self.intensity}"
        )

        observation = np.array([0.0], dtype=np.float32)
        info = {}
        return observation, info

    def render(self, mode: str = "human") -> None:
        print(self.text)
        logger.info(f"Rendering current text: {self.text}")


# Simulate the RL agent interacting with the Hugging Face environment
def main() -> None:
    logger.add(
        "gpt2_training_log.log", rotation="500 MB", level="DEBUG"
    )
    logger.info("Starting GPT2TextGenerationEnv simulation...")

    env = GPT2TextGenerationEnv(max_steps=50)
    model = PPO("MlpPolicy", env, verbose=1)

    logger.info(
        "Starting PPO learning process with 10,000 timesteps..."
    )
    model.learn(total_timesteps=10000)
    logger.info("PPO learning complete.")

    obs, info = env.reset()
    for i in range(50):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            logger.info(f"Episode finished after {i + 1} steps.")
            break


# if __name__ == "__main__":
#     main()
