# CarRacing-v2 Reinforcement Learning Agent
## Overview
This repository contains the implementation of a reinforcement learning agent designed to solve the "CarRacing-v2" environment from the Farama Foundation. The solution is based on a modified Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm, integrated with a pre-trained ResNet18 encoder.
## Features
- Implementation of the TD3 algorithm for continuous action spaces.
- Usage of a pre-trained ResNet18 encoder for efficient feature extraction.
- Epsilon-greedy exploration combined with Ornstein-Uhlenbeck noise for balanced exploration and exploitation.
- The addition of the online-target concept, normally found in Double Q learning 
## More details
A more detailed description of the solution and the decision making behind it can be found in the ProiectATNN.pdf file
## Evaluation metrics
The cumulative reward obtained during an episode of length 1000
## My results
Even though the current result are poor (around -25, SOTA around 900) it shows promissing behaviour (some ability to follow the road and the curves with some small amount of noise added) and it would benefit from more training (the training for this result was done on 500 episodes of length 500).
## Example video link
https://youtu.be/P5MwveUeGu8
