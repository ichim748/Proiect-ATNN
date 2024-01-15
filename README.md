# CarRacing-v2 Reinforcement Learning Agent
## Overview
This repository contains the implementation of a reinforcement learning agent designed to solve the "CarRacing-v2" environment from the Farama Foundation. The solution is based on a modified Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm, integrated with a pre-trained ResNet18 encoder.
## Features
- Implementation of the TD3 algorithm for continuous action spaces.
- Usage of a pre-trained ResNet18 encoder for efficient feature extraction.
- Epsilon-greedy exploration combined with Ornstein-Uhlenbeck noise for balanced exploration and exploitation.
- The addition of the online-target concept, normally found in Double Q learning 
