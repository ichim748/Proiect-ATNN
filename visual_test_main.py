import gymnasium as gym
import cv2
import numpy as np
import torch
from Agent import Agent


if __name__ == '__main__':
    env = gym.make("CarRacing-v2", domain_randomize=False, render_mode='human')
    output_size = 3
    agent = Agent(output_size)
    num_episodes = 1
    max_steps_per_episode = 1000
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps_per_episode):
            action = agent.select_action_pretrained(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if step % 10 == 0:
                print(f"Step {step}:")
                print(action)
                print(reward)
            state = next_state
            total_reward += reward
            if terminated:
                break
        print(f"Episode: {episode}, Total Reward: {total_reward}")
    env.close()
