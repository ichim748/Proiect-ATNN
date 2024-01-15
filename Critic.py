import numpy
import torch
import torch.nn as nn
import torchvision.models
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, fine_tuning=True, fine_tune_layers=50):
        super(CriticNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.fine_tuning = fine_tuning

        self.encoder = torchvision.models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()

        if self.fine_tuning:
            layer = 0
            for param in self.encoder.parameters():
                param.requires_grad = False
                if layer >= fine_tune_layers:
                    param.requires_grad = True
                layer += 1

        self.layer1 = nn.Linear(515, 300)
        self.layer2 = nn.Linear(300, 100)
        self.layer3 = nn.Linear(100, 1)

        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': learning_rate * 0.1},
            {'params': self.layer1.parameters()},
            {'params': self.layer2.parameters()},
            {'params': self.layer3.parameters()}
        ], lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        if not isinstance(state, numpy.ndarray) and torch.is_tensor(state):
            state = state.cpu().numpy()
        if len(state.shape) == 3:
            state = np.transpose(state, (2, 0, 1))
            state = np.expand_dims(state, axis=0)
        else:
            state = np.transpose(state, (0, 3, 1, 2))
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action = torch.tensor(action, device=self.device, dtype=torch.float32)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        encoded_state = torch.cat([self.encoder(state), action], 1)
        x = torch.nn.functional.relu(self.layer1(encoded_state))
        x = torch.nn.functional.relu(self.layer2(x))
        actions = self.layer3(x)
        return actions
