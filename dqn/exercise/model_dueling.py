import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.fc0 = nn.Linear(state_size, 256)
        self.fc1_s = nn.Linear(256, 128)
        self.fc1_a = nn.Linear(256, 128)
        self.fc2_s = nn.Linear(128, 1)
        self.fc2_a = nn.Linear(128, action_size)

        self.hidden_activation = nn.ReLU()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.hidden_activation(self.fc0(state))
        state_values = self.fc2_s(self.hidden_activation(self.fc1_s(x)))
        advantage_values = self.fc2_a(self.hidden_activation(self.fc1_a(x)))
        q = state_values + advantage_values - advantage_values.mean(dim=1, keepdim=True)

        return q
