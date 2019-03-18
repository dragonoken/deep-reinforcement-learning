import torch
import torch.nn as nn
import torch.nn.functional as F

from default_hyperparameters import SEED

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=SEED, **kwds):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        if kwds != {}:
            print("Ignored keyword arguments: ", end='')
            print(*kwds, sep=', ')
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

        gain = nn.init.calculate_gain('relu')
        for module in self.children():
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
                nn.init.xavier_normal_(module.weight.data, gain=gain)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        x = self.hidden_activation(self.fc0(x))
        state_value = self.hidden_activation(self.fc1_s(x))
        state_value = self.fc2_s(state_value)
        advantage_values = self.hidden_activation(self.fc1_a(x))
        advantage_values = self.fc2_a(advantage_values)
        q = state_value + advantage_values - advantage_values.mean(dim=1, keepdim=True)

        return q
