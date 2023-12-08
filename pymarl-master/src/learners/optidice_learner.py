import copy
from components.episode_buffer import EpisodeBatch

import torch.nn.functional as F
import torch as th
from torch.distributions import Uniform
from torch.optim import RMSprop, Adam
import numpy as np

from modules.critics.optidice import OptiDiceCritic


class OptiDiceLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.scheme = scheme
        self.alpha_start = args.alpha_start
        self.alpha_end = args.alpha_end
        self.nu_grad_penalty_coeff = args.nu_grad_penalty_coeff

        self.logger = logger

        self.nu_network = OptiDiceCritic(scheme, 1, args)
        self.nu_optimizer = Adam(params=self.nu_network.parameters(), lr=args.nu_lr)

        policy_params = [param for params in self.mac.parameters() for param in params]
        self.policy_optimizer = Adam(params=policy_params, lr=args.lr)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        assert 'rnn' not in self.args.agent
        self.train_step(batch, t_env)

    def train_step(self, batch: EpisodeBatch, t_env: int):
        alpha = self.alpha_start * (self.alpha_end / self.alpha_start) ** (t_env / self.args.t_max)  # alpha_start -> alpha_end

        seq_len = batch['state'].shape[1] - 1
        states = batch['state'][:, :-1]  # [batch_size, seq_len, state_dim]
        rewards = batch['reward'][:, :-1]  # [batch_size, seq_len, 1]
        terminals = batch['terminated'][:, :-1]  # [batch_size, seq_len, 1]
        next_states = batch['state'][:, 1:]  # [batch_size, seq_len, state_dim]
        obss = batch['obs'][:, :-1]  # [batch_size, seq_len, n_agents, obs_dim]
        actions_onehot = batch['actions_onehot'][:, :-1]  # [batch_size, seq_len, n_agents, n_actions]
        mask = batch['filled'][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminals[:, :-1])
        mask_elems = mask.sum().item()

        nu = self.nu_network(states)  # [batch_size, seq_len, 1]
        next_nu = self.nu_network(next_states)  # [batch_size, seq_len, 1]
        init_nu = self.nu_network(states[:, 0, :])  # [batch_size, 1]

        # nu training
        if self.args.nu_grad_penalty_coeff > 0:
            states1 = states[:len(states)//2]  # [batch_size, seq_len, state_dim]
            states2 = states[len(states)//2:]  # [batch_size, seq_len, state_dim]
            states2 = states2[:, th.randperm(states2.size(1))]  # shuffle across timesteps
            epsilon = th.rand(states1.shape[0], states1.shape[1], 1, device=states.device)
            states_inter = epsilon * states1 + (1 - epsilon) * states2
            states_inter.requires_grad = True
            nu_inter = self.nu_network(states_inter)
            grads_inter = th.autograd.grad(outputs=nu_inter, inputs=states_inter, grad_outputs=th.ones_like(nu_inter), retain_graph=True, create_graph=True, only_inputs=True)[0]
            nu_grad_penalty = (th.norm(grads_inter, dim=-1) ** 2).mean()

        e = rewards + self.args.gamma * next_nu - nu  # [batch_size, seq_len, 1]
        nu_loss = alpha * th.logsumexp(
            (1 - mask) * (-1e10) + e / alpha - np.log(mask_elems),
            dim=(0, 1, 2)) + (1 - self.args.gamma) * init_nu.mean()
        if self.args.nu_grad_penalty_coeff > 0:
            nu_loss += self.nu_grad_penalty_coeff * nu_grad_penalty
        self.nu_optimizer.zero_grad()
        nu_loss.backward()
        self.nu_optimizer.step()

        # Policy extraction
        w = th.exp((e - e.max().detach()) / alpha)
        w = (w / w.mean()).detach()  # [batch_size, seq_len, 1]
        policy_probs = self.mac.forward_policy(batch, seq_len)
        policy_log_probs_data = (th.log(policy_probs + 1e-10) * actions_onehot).sum(dim=-1, keepdims=True).sum(dim=2)  # [batch_size, seq_len, 1]
        policy_loss = -(mask * w * policy_log_probs_data).sum() / mask_elems
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # Logging
            self.logger.log_stat('alpha', alpha, t_env, to_sacred=False)
            self.logger.log_stat('policy_loss', policy_loss.item(), t_env, to_sacred=False)
            self.logger.log_stat(f'nu_loss', nu_loss.item(), t_env, to_sacred=False)
            self.logger.log_stat(f'nu', nu.mean().item(), t_env, to_sacred=False)
            self.logger.log_stat(f'w', w.mean().item(), t_env, to_sacred=False)
            self.logger.log_stat(f'w_max', w.max().item(), t_env, to_sacred=False)
            if 'gfootball' not in self.args.env and self.args.env != "warehouse" and 'sc2' not in self.args.env:
                for x in range(self.n_agents):
                    for y in range(self.n_actions):
                        output_to_info_json = ("mmdp" in self.args.env)
                        self.logger.log_stat(f'policy_agent_{x}_action_{y}', policy_probs[:, 0, x, y].mean().item(), t_env,
                                             to_sacred=output_to_info_json)

            self.log_stats_t = t_env
    def cuda(self):
        self.mac.cuda()
        self.nu_network.cuda()

    def save_models(self, path, training_state):
        self.mac.save_models(path)
        th.save(self.nu_network.state_dict(), f"{path}/nu_network.th")
        th.save(self.nu_optimizer.state_dict(), f"{path}/nu_optimizer.th")
        th.save(self.policy_optimizer.state_dict(), f"{path}/policy_optimizer.th")
        th.save(training_state, f"{path}/training_state.th")

    def load_models(self, path):
        self.mac.load_models(path)
        self.nu_network.load_state_dict(th.load(f"{path}/nu_network.th", map_location=lambda storage, loc: storage))
        self.nu_optimizer.load_state_dict(th.load(f"{path}/nu_optimizer.th", map_location=lambda storage, loc: storage))
        self.policy_optimizer.load_state_dict(th.load(f"{path}/policy_optimizer.th", map_location=lambda storage, loc: storage))
        training_state = th.load(f"{path}/training_state.th")
        return training_state
