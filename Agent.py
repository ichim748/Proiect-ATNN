import torch
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from Actor import ActorNetwork
from Critic import CriticNetwork
from OrnsteinUhlenbeckNoise import OrnsteinUhlenbeckNoise

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'terminated', 'shaped_reward'))


class Agent:
    def __init__(self, policy_update_frequency=4, learning_rate=1e-2, gamma=0.9,
                 tau=1e-2, buffer_size=10000, batch_size=32):
        self.actor_network = ActorNetwork(learning_rate)
        self.target_actor_network = ActorNetwork(learning_rate)
        self.critic_network_1 = CriticNetwork(learning_rate)
        self.target_critic_network_1 = CriticNetwork(learning_rate)
        self.critic_network_2 = CriticNetwork(learning_rate)
        self.target_critic_network_2 = CriticNetwork(learning_rate)

        # use the weights obtained from a previous run
        self.actor_network.load_state_dict(torch.load('actor_final_weights.onnx'))
        self.critic_network_1.load_state_dict(torch.load('critic1_final_weights.onnx'))
        self.critic_network_2.load_state_dict(torch.load('critic2_final_weights.onnx'))
        print("Loaded the pretrained weights!")

        self.target_actor_network.load_state_dict(self.actor_network.state_dict())
        self.target_critic_network_1.load_state_dict(self.critic_network_1.state_dict())
        self.target_critic_network_2.load_state_dict(self.critic_network_2.state_dict())

        self.target_actor_network.eval()
        self.target_critic_network_1.eval()
        self.target_critic_network_1.eval()

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_network_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_network_2.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.tau = tau
        self.step_counter = 0
        self.policy_update_frequency = policy_update_frequency
        self.consecutive_bad_rewards = 0
        self.ou_noise = OrnsteinUhlenbeckNoise(size=3)

    def discretize_action(self, action):
        steering, gas, breaking = action[0], action[1], action[2]
        steering_thresholds = [-0.8, -0.3, 0.3, 0.8]
        throttle_thresholds = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
        braking_thresholds = [0, 0.2, 0.5, 0.8]
        final_outcome = 0
        for idx, i in enumerate(steering_thresholds):
            if steering < i:
                final_outcome += idx
                break
            final_outcome += len(steering_thresholds)
        for idx, i in enumerate(throttle_thresholds):
            if gas < i:
                final_outcome += (idx*10)
                break
            final_outcome += (len(throttle_thresholds)*10)
        for idx, i in enumerate(braking_thresholds):
            if breaking < i:
                final_outcome += (idx*100)
                break
            final_outcome += (len(braking_thresholds) * 100)
        return final_outcome

    def dediscretize_action(self, discretized_action):
        steering_thresholds = [-0.8, -0.3, 0.3, 0.8]
        throttle_thresholds = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
        braking_thresholds = [0, 0.2, 0.5, 0.8]
        final_outcome = []
        steering = discretized_action % 10
        throttle = discretized_action // 10 % 10
        brake = discretized_action // 10 % 10
        if steering < len(steering_thresholds):
            final_outcome.append(steering_thresholds[steering])
        else:
            final_outcome.append(steering_thresholds[len(steering_thresholds)-1])
        if throttle < len(throttle_thresholds):
            final_outcome.append(throttle_thresholds[throttle])
        else:
            final_outcome.append(throttle_thresholds[len(throttle_thresholds)-1])
        if brake < len(braking_thresholds):
            final_outcome.append(braking_thresholds[brake])
        else:
            final_outcome.append(braking_thresholds[len(braking_thresholds)-1])
        return np.array(final_outcome)

    def reset(self):
        self.ou_noise.reset()

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            steering = np.random.uniform(-1, 1)
            throttle = np.random.uniform(0, 1)
            brake = np.random.uniform(0, 1)
            action = np.array([steering, throttle, brake])
        else:
            with torch.no_grad():
                filtered_state = [t for t in state if len(t) > 1 and type(t) is not tuple]
                state = torch.tensor(np.array(filtered_state), device=self.actor_network.device, dtype=torch.float32)
                action = self.actor_network(state).cpu().numpy()
                noise = self.ou_noise.sample()
                action = action + noise
                # action = np.clip(action, -1.0, 1.0)
                action = action.squeeze()
                action[0] = np.clip(action[0], -1.0, 1.0)
                action[1] = np.clip(action[1], 0, 1.0)
                action[2] = np.clip(action[2], 0, 1.0)
        return action

    def select_action_pretrained(self, state):
        with torch.no_grad():
            filtered_state = [t for t in state if len(t) > 1 and type(t) is not tuple]
            state = torch.tensor(np.array(filtered_state), device=self.actor_network.device, dtype=torch.float32)
            action = self.actor_network(state).cpu().numpy()
            # action = np.clip(action, -1.0, 1.0)
            noise = self.ou_noise.sample()
            action = action + noise
            action = action.squeeze()
            action[0] = np.clip(action[0], -1.0, 1.0)
            action[1] = np.clip(action[1], 0, 1.0)
            action[2] = np.clip(action[2], 0, 1.0)
        return action

    def up_to_date_reward(self, reward):
        if reward <= 0:
            self.consecutive_bad_rewards += 1
        else:
            self.consecutive_bad_rewards -= 10
        output = -10 * self.consecutive_bad_rewards + reward
        return output

    def reset_consecutive_bad_rewards(self):
        self.consecutive_bad_rewards = 0

    def step(self, state, action, next_state, reward, terminated, truncated, info):
        updated_reward = self.up_to_date_reward(reward)
        experience = Experience(state, action, reward, next_state, terminated, updated_reward)
        self.buffer.append(experience)
        if len(self.buffer) > self.batch_size:
            self.learn()

    def learn(self):
        batch = random.sample(self.buffer, self.batch_size)
        batch = [exp for exp in batch if type(exp.state) is not tuple and len(exp.state) > 0
                 and type(exp.next_state) is not tuple and len(exp.next_state) > 0]
        states, actions, rewards, next_states, dones, updated_rewards = zip(*batch)

        states = torch.FloatTensor(states).to(self.actor_network.device)
        actions = torch.FloatTensor(actions).to(self.actor_network.device)
        rewards = torch.FloatTensor(rewards).to(self.actor_network.device)
        next_states = torch.FloatTensor(next_states).to(self.actor_network.device)
        dones = torch.FloatTensor(dones).to(self.actor_network.device)
        updated_rewards = torch.FloatTensor(updated_rewards).to(self.actor_network.device).view(-1, 1)

        with torch.no_grad():
            next_actions = self.target_actor_network(next_states)
            target_Q1 = self.target_critic_network_1(next_states, next_actions)
            target_Q2 = self.target_critic_network_2(next_states, next_actions)

            rewards = rewards.view(-1, 1)
            dones = dones.view(-1, 1)
            target_Q = updated_rewards + ((1 - dones) * self.gamma * torch.min(target_Q1, target_Q2)).detach()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        current_Q1 = self.critic_network_1(states, actions)
        current_Q2 = self.critic_network_2(states, actions)
        loss_critic_1 = torch.nn.functional.mse_loss(current_Q1, target_Q)
        loss_critic_2 = torch.nn.functional.mse_loss(current_Q2, target_Q)
        loss_critic_1.backward()
        loss_critic_2.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.step_counter % self.policy_update_frequency == 0:
            self.actor_optimizer.zero_grad()
            policy_loss = -self.critic_network_1(states, self.actor_network(states)).mean()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.target_critic_network_1, self.critic_network_1)
            self.soft_update(self.target_critic_network_2, self.critic_network_2)
            self.soft_update(self.target_actor_network, self.actor_network)

        self.step_counter += 1

    def soft_update(self, target_network, online_network):
        with torch.no_grad():
            for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
                updated_weight = self.tau * online_param.data + (1 - self.tau) * target_param.data
                target_param.data.copy_(updated_weight)
