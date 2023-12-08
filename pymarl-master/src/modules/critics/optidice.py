import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Mapping


class OptiDiceCritic(nn.Module):
    def __init__(self, scheme: Mapping, output_dim: int, args, is_state_action_input: bool = False, agent_id: Optional[int] = None, last_layer_bias: bool = False):
        super(OptiDiceCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.agent_id = agent_id

        input_shape = self._get_input_shape(scheme, is_state_action_input)

        self.tabular_network = self.args.tabular_network
        if self.tabular_network:
            self.tabular_fc = nn.Linear(input_shape, output_dim, bias=last_layer_bias)
        else:
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc3 = nn.Linear(args.rnn_hidden_dim, output_dim, bias=last_layer_bias)

    def forward(self, states, t=None):
        if self.tabular_network:
            x = self.tabular_fc(states)
        else:
            x = F.relu(self.fc1(states))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x

    def _get_input_shape(self, scheme, is_state_action_input):
        if is_state_action_input:
            return scheme["state"]["vshape"] + scheme["actions_onehot"]["vshape"][0]
        else:
            return scheme["state"]["vshape"]
