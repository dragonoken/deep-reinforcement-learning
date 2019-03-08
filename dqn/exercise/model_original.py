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
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, action_size)
        
        self.hidden_activation = nn.ReLU()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc0(state)
        x = self.hidden_activation(x)
        x = self.fc1(x)
        x = self.hidden_activation(x)
        x = self.fc2(x)
        
        return x
