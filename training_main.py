import gymnasium as gym
import cv2
import numpy as np
import torch
from Agent import Agent


if __name__ == '__main__':
    env = gym.make("CarRacing-v2", domain_randomize=False)
    agent = Agent()

    num_episodes = 500
    max_steps_per_episode = 500
    epsilon = 0.5
    original_epsilon = 0.5

    for episode in range(num_episodes):
        state = env.reset()
        agent.reset()
        total_reward = 0
        agent.reset_consecutive_bad_rewards()
        for step in range(max_steps_per_episode):
            action = agent.select_action(state, epsilon)
            epsilon -= (original_epsilon / num_episodes * 0.3 * max_steps_per_episode)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.step(state, action, next_state, reward, terminated, truncated, info)
            state = next_state
            total_reward += reward
            if terminated:
                break
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        if episode % 10 == 0:
            torch.save(agent.actor_network.state_dict(), f'tempActorWeights\\temp_weights_episode_{episode}.onnx')
            torch.save(agent.critic_network_1.state_dict(), f'tempCritic1Weights\\temp_weights_episode_{episode}.onnx')
            torch.save(agent.critic_network_2.state_dict(), f'tempCritic2Weights\\temp_weights_episode_{episode}.onnx')
    torch.save(agent.actor_network.state_dict(), 'actor_final_weights.onnx')
    torch.save(agent.critic_network_1.state_dict(), 'critic1_final_weights.onnx')
    torch.save(agent.critic_network_2.state_dict(), 'critic2_final_weights.onnx')
    env.close()
