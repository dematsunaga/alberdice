REGISTRY = {}

from .rnn_agent import RNNAgent
from .mlp_agent import MLPAgent
from.sep_mlp_agent import SepMLPAgent
from .sep_rnn_agent import SepRNNAgent
from .sep_transformer_agent import SepTransformerAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["sep_mlp"] = SepMLPAgent
REGISTRY["sep_rnn"] = SepRNNAgent
REGISTRY["sep_transformer"] = SepTransformerAgent
REGISTRY["transformer"] = SepTransformerAgent
