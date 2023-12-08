import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
import math
"""
Adapted from Multi-Agent Transformers
"""
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_timesteps, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", th.tril(th.ones(n_timesteps + 2, n_timesteps + 2))
                             .view(1, 1, n_timesteps + 2, n_timesteps + 2))

        self.att_bp = None

    def forward(self, key, value, query):
        # B: batch size
        # T: timesteps
        # D: embedding dimension = 64
        B, T, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, T, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(query).view(B, T, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(value).view(B, T, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y # B, T, D


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_timesteps):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_timesteps, masked=True)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x

class Encoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_timesteps, add_actions, device):
        super(Encoder, self).__init__()

        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_timesteps = n_timesteps
        self.device = device
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.add_actions = add_actions
        if add_actions:
            self.action_encoder = nn.Sequential(nn.LayerNorm(action_dim),
                                             init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_timesteps) for _ in range(n_block)])

        self.embed_timestep = nn.Embedding(n_timesteps + 1, n_embd)

    def forward(self, obs, actions):
        # obs: (batch, n_agent, obs_dim)

        timesteps = th.arange(0, obs.shape[1], device=obs.device, dtype=th.long)
        timestep_embeddings = self.embed_timestep(timesteps)  # [seq_len, n_embd]

        obs_embeddings = self.obs_encoder(obs) + timestep_embeddings  # [batch_size, seq_len, n_embed]
        if self.add_actions:
            x = th.zeros(obs_embeddings.shape[0], obs_embeddings.shape[1] * 2, obs_embeddings.shape[2])
            x[:, ::2, :] = obs_embeddings
            if actions.shape[1] > 0:
                action_embeddings = self.action_encoder(actions) + timestep_embeddings  # [batch_size, seq_len, n_embed]
                x[:, 1::2, :] = action_embeddings
        else:
            x = obs_embeddings

        x = x.to(self.device)
        rep = self.blocks(self.ln(x))

        return rep

class SepTransformerAgent(nn.Module):
    def __init__(self, input_shape, args, id=None):
        super(SepTransformerAgent, self).__init__()
        self.args = args
        self.id = id

        n_embed = self.args.n_embed
        n_block = self.args.n_block
        n_head = self.args.n_head
        action_dim = self.args.n_actions
        if self.args.add_actions_transformer:
            n_timesteps = self.args.env_args["episode_limit"] * 2
        else:
            n_timesteps = self.args.env_args["episode_limit"]
        self.encoder = Encoder(input_shape, action_dim, n_block, n_embed, n_head, n_timesteps,
                               add_actions=self.args.add_actions_transformer, device=self.args.device)

        self.head = nn.Sequential(init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU(), nn.LayerNorm(n_embed),
                                  init_(nn.Linear(n_embed, action_dim)))
        self.init_weights()

    def forward(self, obs, actions_onehot, hidden_state=None):
        obs_rep = self.encoder(obs, actions_onehot)
        pi = self.head(obs_rep)
        if self.args.add_actions_transformer:
            return pi[:, ::2], hidden_state
        else:
            return pi, hidden_state

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)
