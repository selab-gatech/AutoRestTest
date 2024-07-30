import torch
import torch.nn as nn

num_batches = 32

# Network for value, data source, and parameter agent -> Gets all Q-values for all possible actions given state
class GeneralNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(GeneralNetwork, self).__init__()
        self.fc1 = nn.Linear(state_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_shape)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Network for dependency agent -> Gets Q-value (singular) for state + action pair
class DependencyNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(DependencyNetwork, self).__init__()
        self.action_fc = nn.Linear(action_shape, 64)
        self.state_fc = nn.Linear(state_shape, 64)
