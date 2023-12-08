import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import MultivariateNormal


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        with th.no_grad():
            th.nn.init.xavier_uniform(self.fc1.weight)
            th.nn.init.xavier_uniform(self.fc2.weight)
            th.nn.init.uniform_(self.fc3.weight, a=1e-3, b=1e-3)

    def forward(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        pi = self.fc3(x)

        return pi, hidden_state  # so we dont need to change mac
