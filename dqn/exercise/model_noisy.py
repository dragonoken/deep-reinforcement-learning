import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from default_hyperparameters import SEED, INIT_SIGMA, LINEAR

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, initial_sigma=INIT_SIGMA):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_sigma = initial_sigma
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noisy_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('noisy_bias', None)
        self.reset_parameters()

        self.noise = True

    def reset_parameters(self):
        bound = math.sqrt(3 / self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.constant_(self.noisy_weight, self.initial_sigma)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
            nn.init.constant_(self.noisy_bias, self.initial_sigma)

    def forward(self, input):
        if self.noise:
            return F.linear(
                       input,
                       self.weight + (self.noisy_weight * torch.randn_like(self.noisy_weight)),
                       self.bias + (self.noisy_bias * torch.randn_like(self.noisy_bias))
                   )
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, linear_type=LINEAR, initial_sigma=INIT_SIGMA, seed=SEED, **kwds):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            linear_type (str): type of linear layers ('linear', 'noisy')
            initial_sigma (float): initial weight value for noise parameters
                when using noisy linear layers
            seed (int): Random seed
        """
        if kwds != {}:
            print("Ignored keyword arguments: ", end='')
            print(*kwds, sep=', ')
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_type = linear_type.lower()
        self.seed = torch.manual_seed(seed)

        def noisy_layer(in_features, out_features):
            return NoisyLinear(in_features, out_features, True, initial_sigma)
        linear = {'linear': nn.Linear, 'noisy': noisy_layer}[self.linear_type]

        self.fc0 = linear(state_size, 256)
        self.fc1_s = linear(256, 128)
        self.fc1_a = linear(256, 128)
        self.fc2_s = linear(128, 1)
        self.fc2_a = linear(128, action_size)

        self.hidden_activation = nn.ReLU()

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

    def noise(self, enable):
        enable = bool(enable)
        for m in self.children():
            if isinstance(m, NoisyLinear):
                m.noise = enable
