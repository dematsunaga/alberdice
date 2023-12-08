import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import Categorical


class SepRNNAgent(nn.Module):
    def __init__(self, input_shape, args, id):
        super(SepRNNAgent, self).__init__()
        self.args = args
        self.id = id

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # initialization
        with th.no_grad():
            th.nn.init.xavier_uniform(self.fc1.weight)
            th.nn.init.xavier_uniform(self.fc2.weight)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

