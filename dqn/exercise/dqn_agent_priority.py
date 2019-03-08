import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate
UPDATE_EVERY = 4        # how often to update the network
A = 0.95                # randomness vs priority
P_EPS = 1e-04           # priority epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, initial_a=A, initial_beta=0.):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.a = initial_a
        self.beta = initial_beta

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, is_weights, indices = self.memory.sample(self.a, self.beta)
                new_priorities = self.learn(experiences, is_weights, GAMMA)
                self.memory.update_priorities(indices, new_priorities)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(np.arange(self.action_size))

    def learn(self, experiences, is_weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            is_weights (torch.Tensor): tensor of importance-sampling weights
            gamma (float): discount factor

        Returns
        =======
            new_priorities (List[float]): list of new priority values for the given sample
        """
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            target = rewards + gamma * (1 - dones) * self.qnetwork_target(next_states)\
                                                         .gather(dim=1, index=self.qnetwork_local(next_states)\
                                                                                  .argmax(dim=1, keepdim=True))

        pred = self.qnetwork_local(states)

        diff = target.sub(pred.gather(dim=1, index=actions))
        new_priorities = diff.detach().abs().add(P_EPS).cpu().numpy().reshape((-1,))
        loss = diff.pow(2).mul(is_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        return new_priorities

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(self.max_priority)

    def sample(self, a, beta):
        """Randomly sample a batch of experiences from memory.

        Params
        ======
            a (int or float): parameter used for calculating sample probabilities
            beta (int or float): parameter used for calculating importance-priority weights

        Returns
        =======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            is_weights (torch.Tensor): tensor of importance-sampling weights
            indices (np.ndarray): sample indices"""
        probs = np.power(np.array(self.priorities), a)
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=probs)

        states, actions, rewards, next_states, dones = zip(*tuple(self.memory[i] for i in indices if self.memory[i] is not None))
        is_weights = tuple(probs[i] for i in indices if self.memory[i] is not None)

        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8).reshape((-1, 1))).float().to(device)

        is_weights = np.power(np.multiply(is_weights, len(self.memory)), -beta)
        is_weights = torch.from_numpy(np.divide(is_weights, max(is_weights)).reshape((-1, 1))).float().to(device)

        return (states, actions, rewards, next_states, dones), is_weights, indices

    def update_priorities(self, indices, new_priorities):
        """Update the priorities for the experiences of given indices to the given new values.

        Params
        ======
            indices (array_like): indices of experience priorities to update
            new_priorities (array-like): new priority values for given indices"""
        for i, new_priority in zip(indices, new_priorities):
            self.priorities[i] = new_priority
        self.max_priority = max(self.priorities)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
