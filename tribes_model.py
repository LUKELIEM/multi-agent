import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Policy(nn.Module):
    # an index parameter is needed to allow for multiple learning agents later
    def __init__(self, observation_size, agent_idx=1):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(observation_size, 128)
        self.affine2 = nn.Linear(128, 8)

        self.saved_actions = []
        self.log_probs = []   # Added to implement REINFORCE for PyTorch 0.4.1
        self.rewards = []
        self.idx = agent_idx   # This allows multiple learning agents

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)  # Added dim=1 to support PyTorch 0.4.1

    # The weights should be allowed to be saved into and load from agent-indexed model files
    # e.g. agent-1-model.pkl, agent-2-model.pkl, etc.
    def save_weights(self):
        file_name = 'agent-'+str(self.idx)+'-model.pkl'
        torch.save(self.state_dict(), file_name)   

    def load_weights(self):
        file_name = 'agent-'+str(self.idx)+'-model.pkl'
        if not os.path.exists(file_name):
                raise ValueError('map not found: ' + file_name)
        self.load_state_dict(torch.load(file_name))

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self(Variable(state))   # This is tricky --> probs = Policy(state)

        # reinforce() was removed in PyTorch 0.4.1
        # Use torch.distributions instead. See http://pytorch.org/docs/master/distributions.html

        # The following code replaces the previous implementation
        m = torch.distributions.Categorical(probs) 
        action = m.sample()
        log_prob = m.log_prob(action)

        # Stack actions and log_probs for REINFORCE update at the end of episode
        self.log_probs.append(log_prob)
        self.saved_actions.append(action)

        return action.data[0], log_prob


# Just a dumb random agent
class Rdn_Policy():
    def __init__(self):
        super(Rdn_Policy, self).__init__()

    def select_action(self, state):
        return random.randrange(0, 8), 0  # return random action, log_prob of zero