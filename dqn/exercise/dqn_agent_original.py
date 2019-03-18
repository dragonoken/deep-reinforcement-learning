import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from default_hyperparameters import SEED, BUFFER_SIZE, BATCH_SIZE, START_SINCE, GAMMA,\
                                    T_UPDATE, TAU, LR, WEIGHT_DECAY, UPDATE_EVERY, CLIP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=SEED, batch_size=BATCH_SIZE,
                 buffer_size=BUFFER_SIZE, start_since=START_SINCE, gamma=GAMMA, target_update_every=T_UPDATE,
                 tau=TAU, lr=LR, weight_decay=WEIGHT_DECAY, update_every=UPDATE_EVERY, clip=CLIP, **kwds):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            batch_size (int): size of each sample batch
            buffer_size (int): size of the experience memory buffer
            start_since (int): number of steps to collect before start training
            gamma (float): discount factor
            target_update_every (int): how often to update the target network
            tau (float): target network soft-update parameter
            lr (float): learning rate
            weight_decay (float): weight decay for optimizer
            update_every (int): update(learning and target update) interval
            clip (float): gradient norm clipping (`None` to disable)
        """
        if kwds != {}:
            print("Ignored keyword arguments: ", end='')
            print(*kwds, sep=', ')
        assert isinstance(state_size, int)
        assert isinstance(action_size, int)
        assert isinstance(seed, int)
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(buffer_size, int) and buffer_size >= batch_size
        assert isinstance(start_since, int) and batch_size <= start_since <= buffer_size
        assert isinstance(gamma, (int, float)) and 0 <= gamma <= 1
        assert isinstance(target_update_every, int) and target_update_every > 0
        assert isinstance(tau, (int, float)) and 0 <= tau <= 1
        assert isinstance(lr, (int, float)) and lr >= 0
        assert isinstance(weight_decay, (int, float)) and weight_decay >= 0
        assert isinstance(update_every, int) and update_every > 0
        if clip: assert isinstance(clip, (int, float)) and clip >= 0

        self.state_size          = state_size
        self.action_size         = action_size
        self.seed                = random.seed(seed)
        self.batch_size          = batch_size
        self.buffer_size         = buffer_size
        self.start_since         = start_since
        self.gamma               = gamma
        self.target_update_every = target_update_every
        self.tau                 = tau
        self.lr                  = lr
        self.weight_decay        = weight_decay
        self.update_every        = update_every

        # Q-Network
        self.qnetwork_local  = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr, weight_decay=weight_decay)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps and TARGET_UPDATE_EVERY steps)
        self.u_step = 0
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.u_step = (self.u_step + 1) % self.update_every
        if self.u_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.start_since:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

        # update the target network every TARGET_UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.target_update_every
        if self.t_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

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

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            target = rewards + gamma * self.qnetwork_target(next_states).max(dim=1)[0] * (1 - dones)

        pred = self.qnetwork_local(states)

        loss = F.mse_loss(pred.gather(dim=1, index=actions), target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), CLIP)
        self.optimizer.step()

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
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None]).reshape((-1, 1))).long().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None]).reshape((-1, 1))).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None], dtype=np.uint8).reshape((-1, 1))).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
