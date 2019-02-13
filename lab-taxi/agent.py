import numpy as np
import sys
from collections import defaultdict

class Agent_A:
    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        self.mode = ['zero', 'max', 'expected'][2]
        
        self.initial_alpha = 0.2
        self.alpha = self.initial_alpha
        self.min_alpha = 0.0001
        self.final_alpha = None
        
        self.alpha_decay_mode = ['linear', 'exponential'][0]
        self.alpha_decay_duration = 100000
        if self.alpha_decay_mode == 'linear':
            self.alpha_decay_rate = (self.min_alpha - self.initial_alpha) / self.alpha_decay_duration
        else:
            self.alpha_decay_factor = (self.min_alpha / self.initial_alpha) ** (1 / self.alpha_decay_duration)
        
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon
        self.min_epsilon = 0.000001
        self.final_epsilon = None
        
        self.epsilon_decay_mode = ['linear', 'exponential'][1]
        self.epsilon_decay_duration = 100000
        if self.epsilon_decay_mode == 'linear':
            self.epsilon_decay_rate = (self.min_epsilon - self.initial_epsilon) / self.epsilon_decay_duration
        else:
            self.epsilon_decay_factor = (self.min_epsilon / self.initial_epsilon) ** (1 / self.epsilon_decay_duration)
        
        self.reached_min_alpha = (self.alpha <= self.min_alpha)
        self.reached_min_epsilon = (self.epsilon <= self.min_epsilon)

        self.gamma = 1.0

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action_values = self.Q[state]
        is_max = np.equal(action_values, action_values.max())
        probs = np.full(self.nA, self.epsilon / self.nA) + is_max * (1 - self.epsilon) / sum(is_max)
        action = np.random.choice(self.nA, p=probs)
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        action_values = self.Q[next_state]
        is_max = np.equal(action_values, action_values.max())
        probs = np.full(self.nA, self.epsilon / self.nA) + is_max * (1 - self.epsilon) / sum(is_max)

        if self.mode == 'zero':
            expected_return = np.random.choice(action_values, p=probs)
        elif self.mode == 'max':
            expected_return = np.max(action_values)
        else:
            expected_return = np.sum(np.multiply(probs, action_values))
        
        self.Q[state][action] += self.alpha * (reward + self.gamma * expected_return - self.Q[state][action])
        
        if done:
            if not self.reached_min_alpha:
                if self.alpha_decay_mode == 'linear':
                    self.alpha += self.alpha_decay_rate
                elif self.alpha_decay_mode == 'exponential':
                    self.alpha *= self.alpha_decay_factor
                else:
                    raise RuntimeError("Invalid Mode: {}".format(self.alpha_decay_mode))
                if self.alpha <= self.min_alpha:
                    self.reached_min_alpha = True
                    if self.final_alpha is None:
                        self.alpha = self.min_alpha
                    else:
                        self.alpha = self.final_alpha
            
            if not self.reached_min_epsilon:
                if self.epsilon_decay_mode == 'linear':
                    self.epsilon += self.epsilon_decay_rate
                elif self.epsilon_decay_mode == 'exponential':
                    self.epsilon *= self.epsilon_decay_factor
                else:
                    raise RuntimeError("Invalid Mode: {}".format(self.epsilon_decay_mode))
                if self.epsilon <= self.min_epsilon:
                    self.reached_min_epsilon = True
                    if self.final_epsilon is None:
                        self.epsilon = self.min_epsilon
                    else:
                        self.epsilon = self.final_epsilon

class Agent_B:
    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        self.initial_alpha = 0.3
        self.alpha = self.initial_alpha
        self.min_alpha = 0.001
        
        self.alpha_decay_mode = ['linear', 'exponential'][0]
        self.alpha_decay_duration = 20000
        if self.alpha_decay_mode == 'linear':
            self.alpha_decay_rate = (self.min_alpha - self.initial_alpha) / self.alpha_decay_duration
        else:
            self.alpha_decay_factor = (self.min_alpha / self.initial_alpha) ** (1 / self.alpha_decay_duration)
        
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_alpha
        self.min_epsilon = 0.00001
        
        self.epsilon_decay_mode = ['linear', 'exponential'][1]
        self.epsilon_decay_duration = 20000
        if self.epsilon_decay_mode == 'linear':
            self.epsilon_decay_rate = (self.min_epsilon - self.initial_epsilon) / self.epsilon_decay_duration
        else:
            self.epsilon_decay_factor = (self.min_epsilon / self.initial_epsilon) ** (1 / self.epsilon_decay_duration)
        
        self.reached_min_alpha = (self.alpha <= self.min_alpha)
        self.reached_min_epsilon = (self.epsilon <= self.min_epsilon)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action_values = self.Q[state]
        is_max = np.equal(action_values, action_values.max())
        probs = np.full(self.nA, self.epsilon / self.nA) + is_max * (1 - self.epsilon) / sum(is_max)
        action = np.random.choice(self.nA, p=probs)
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        action_values = self.Q[next_state]
        is_max = np.equal(action_values, action_values.max())
        probs = np.full(self.nA, self.epsilon / self.nA) + is_max * (1 - self.epsilon) / sum(is_max)
        
        self.Q[state][action] += self.alpha * (reward + np.sum(probs * action_values) - self.Q[state][action])
        
        if done:
            if not self.reached_min_alpha:
                if self.alpha_decay_mode == 'linear':
                    self.alpha += self.alpha_decay_rate
                elif self.alpha_decay_mode == 'exponential':
                    self.alpha *= self.alpha_decay_factor
                else:
                    raise RuntimeError("Invalid Mode: {}".format(self.alpha_decay_mode))
                if self.alpha <= self.min_alpha:
                    self.reached_min_alpha = True
                    self.alpha = self.min_alpha
            
            if not self.reached_min_epsilon:
                if self.epsilon_decay_mode == 'linear':
                    self.epsilon += self.epsilon_decay_rate
                elif self.epsilon_decay_mode == 'exponential':
                    self.epsilon *= self.epsilon_decay_factor
                else:
                    raise RuntimeError("Invalid Mode: {}".format(self.epsilon_decay_mode))
                if self.epsilon <= self.min_epsilon:
                    self.reached_min_epsilon = True
                    self.epsilon = self.min_epsilon

Agent = Agent_A
