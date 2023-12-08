import copy
from components.episode_buffer import EpisodeBatch
from controllers.basic_controller import BasicMAC

import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import Categorical
from torch.optim import RMSprop, Adam
import numpy as np
from typing import Callable

from modules.critics.optidice import OptiDiceCritic


class AlberDiceLearner:
    def __init__(self, mac: BasicMAC, data_mac: BasicMAC, ar_data_mac: BasicMAC, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents  # the number of agents
        self.n_actions = args.n_actions  # the number of actions for each agent
        self.mac = mac
        self.data_mac = data_mac
        self.ar_data_mac = ar_data_mac
        self.scheme = scheme
        self.alpha_start = args.alpha_start
        self.alpha_end = args.alpha_end
        self.nu_grad_penalty_coeff = args.nu_grad_penalty_coeff

        self.logger = logger

        self.nu_networks = [OptiDiceCritic(scheme, 1, args, agent_id=i) for i in range(self.n_agents)]
        self.nu_optimizers = [Adam(params=v.parameters(), lr=args.nu_lr) for v in self.nu_networks]
        self.e_networks = [OptiDiceCritic(scheme, 1, args, agent_id=i, is_state_action_input=True, last_layer_bias=True) for i in range(self.n_agents)]
        self.e_optimizers = [Adam(params=e.parameters(), lr=args.nu_lr) for e in self.e_networks]
        self.data_policy_optimizer = Adam(params=self.data_mac.agent.parameters(), lr=args.lr)
        if 'sep' in self.args.agent:
            self.policy_optimizers = [Adam(params=agent.parameters(), lr=args.lr) for agent in self.mac.agent]
            self.ar_data_policy_optimizers = [Adam(params=agent.parameters(), lr=args.lr) for agent in self.ar_data_mac.agent]
        else:
            self.policy_optimizer = Adam(params=self.mac.agent.parameters(), lr=args.lr)
            self.ar_data_policy_optimizer = Adam(params=self.ar_data_mac.agent.parameters(), lr=args.lr)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, sample_batch: Callable[[], EpisodeBatch], t_env: int, episode_num: int, show_demo=False, save_data=None):
        assert 'rnn' not in self.args.agent  # TODO: currently assume MLP agent
        agent_order = [i for i in range(self.n_agents)]

        batches = [sample_batch() for _ in range(self.args.num_iter_inner_loop)]
        for agent_id in agent_order:
            for loop_idx in range(self.args.num_iter_inner_loop):
                batch = batches[loop_idx]
                self.train_step(batch, t_env, agent_id)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.log_stats_t = t_env

    def train_step(self, batch: EpisodeBatch, t_env: int, agent_id: int):
        alpha = self.alpha_start * (self.alpha_end / self.alpha_start) ** (t_env / self.args.t_max)  # alpha_start -> alpha_end

        seq_len = batch['state'].shape[1] - 1
        states = batch['state'][:, :-1]  # [batch_size, seq_len, state_dim]
        rewards = batch['reward'][:, :-1]  # [batch_size, seq_len, 1]
        terminals = batch['terminated'][:, :-1]  # [batch_size, seq_len, 1]
        next_states = batch['state'][:, 1:]  # [batch_size, seq_len, state_dim]
        actions_onehot = batch['actions_onehot'][:, :-1]  # [batch_size, seq_len, n_agents, n_actions]
        mask = batch['filled'][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminals[:, :-1])
        mask_elems = mask.sum().item()

        nu_i = self.nu_networks[agent_id](states)  # [batch_size, seq_len, 1]
        next_nu_i = self.nu_networks[agent_id](next_states)  # [batch_size, seq_len, 1]
        init_nu_i = self.nu_networks[agent_id](states[:, :, :])  # [batch_size, seq_len, 1]

        # policy probabilities
        policy_probs = self.mac.forward_policy(batch, seq_len, agent_id=agent_id)
        ar_data_policy_probs = self.ar_data_mac.forward_ar_data_policy(batch, seq_len, agent_id) + 1e-10
        other_policy_probs = policy_probs[:, :, th.arange(self.n_agents) != agent_id]
        other_actions = actions_onehot[:, :, th.arange(self.n_agents) != agent_id]

        # data policy training
        if agent_id == 0:
            data_policy_probs = self.data_mac.forward_policy(batch, seq_len) + 1e-10
            data_policy_loss = -(th.log((data_policy_probs * actions_onehot).sum(dim=-1))).mean()
            self.data_policy_optimizer.zero_grad()
            data_policy_loss.backward()
            self.data_policy_optimizer.step()
        else:
            data_policy_probs = self.data_mac.forward_policy(batch, seq_len, test_mode=True) + 1e-10

        # autoregressive data policy training
        ar_data_policy_loss = -(th.log(ar_data_policy_probs) * other_actions).sum(dim=-1).mean()
        if 'sep' in self.args.agent:
            self.ar_data_policy_optimizers[agent_id].zero_grad()
            ar_data_policy_loss.backward()
            self.ar_data_policy_optimizers[agent_id].step()
        else:
            self.ar_data_policy_optimizer.zero_grad()
            ar_data_policy_loss.backward()
            self.ar_data_policy_optimizer.step()

        # nu training
        if self.args.nu_grad_penalty_coeff > 0:
            states1 = states[:len(states)//2]  # [batch_size, seq_len, state_dim]
            states2 = states[len(states)//2:]  # [batch_size, seq_len, state_dim]
            states2 = states2[:, th.randperm(states2.size(1))]  # shuffle across timesteps
            epsilon = th.rand(states1.shape[0], states1.shape[1], 1, device=states.device)
            states_inter = epsilon * states1 + (1 - epsilon) * states2
            states_inter.requires_grad = True
            nu_inter = self.nu_networks[agent_id](states_inter)
            grads_inter = th.autograd.grad(outputs=nu_inter, inputs=states_inter, grad_outputs=th.ones_like(nu_inter), retain_graph=True, create_graph=True, only_inputs=True)[0]
            nu_grad_norm = th.norm(grads_inter, dim=-1)
            nu_grad_penalty = (nu_grad_norm ** 2).mean()

        other_policy_dist = Categorical(other_policy_probs)
        other_policy_actions = F.one_hot(other_policy_dist.sample((16,)), num_classes=self.n_actions)  # [16, batch_size, seq_len, n_agents-1, n_actions]
        other_policy_kl = th.log(((other_policy_probs / ar_data_policy_probs) * other_policy_actions).sum(dim=-1)).mean(dim=0).sum(dim=-1, keepdims=True).detach()  # [batch_size, seq_len, 1]

        e_i = rewards * self.args.reward_weight - alpha * other_policy_kl + self.args.gamma * next_nu_i - nu_i  # [batch_size, seq_len, 1]

        # resample data policy
        other_policy_ratio = th.prod(
            ((other_policy_probs / ar_data_policy_probs) * other_actions).sum(dim=-1), # [batch_size, seq_len, n_agents-1]
            dim=-1, keepdims=True).detach()  # [batch_size, seq_len, 1]
        other_policy_ratio = th.clamp(other_policy_ratio, max=self.args.other_policy_ratio_max)

        resampling_probs = (other_policy_ratio / (other_policy_ratio.sum() + 1e-10)).flatten()  # [batch_size * seq_len * 1]
        sample_indices = Categorical(resampling_probs).sample(resampling_probs.shape)
        other_policy_ratio_mean = other_policy_ratio.mean() + 1e-10

        nu_loss = alpha * th.logsumexp(
            th.log(other_policy_ratio_mean) + e_i.flatten()[sample_indices] / alpha - np.log(mask_elems),
            dim=0) + (1 - self.args.gamma) * init_nu_i.mean()
        if self.args.nu_grad_penalty_coeff > 0:
            nu_loss += self.nu_grad_penalty_coeff * nu_grad_penalty
        self.nu_optimizers[agent_id].zero_grad()
        nu_loss.backward()
        self.nu_optimizers[agent_id].step()

        # e-network training
        all_actions_onehot_i = th.eye(self.n_actions).repeat(states.shape[:2] + (1, 1)).to(states.device)  # [batch_size, seq_len, n_actions, n_actions]
        all_state_actions_i = th.zeros(
            (self.args.batch_size, seq_len, self.n_actions, states.shape[-1] + self.n_actions)).to(states.device)
        all_state_actions_i[:, :, :, :states.shape[-1]] = states.unsqueeze(2).repeat(1, 1, self.n_actions, 1)
        all_state_actions_i[:, :, :, states.shape[-1]:] = all_actions_onehot_i
        e_i_all_actions = self.e_networks[agent_id](all_state_actions_i)[:, :, :, 0]  # [batch_size, seq_len, n_actions]
        e_i_pred = (e_i_all_actions * actions_onehot[:, :, agent_id, :]).sum(dim=-1, keepdims=True)  # [batch_size, seq_len, 1]

        e_i_loss = ((e_i_pred - e_i.detach()) ** 2).flatten()[sample_indices].mean()  # regression
        e_i_loss += self.args.cql_coeff * (th.logsumexp(e_i_all_actions, dim=-1, keepdims=True) - (e_i_all_actions * data_policy_probs[:, :, agent_id, :].detach()).sum(dim=-1, keepdims=True)).flatten()[sample_indices].mean()  # conservative loss (data policy)

        self.e_optimizers[agent_id].zero_grad()
        e_i_loss.backward()
        self.e_optimizers[agent_id].step()

        # i-projection
        policy_loss = -(policy_probs[:, :, agent_id] * e_i_all_actions.detach()).sum(dim=-1).flatten()[
            sample_indices].mean()
        policy_loss += alpha * (policy_probs[:, :, agent_id] * (
                    th.log(policy_probs[:, :, agent_id] + 1e-10) - th.log(
                data_policy_probs[:, :, agent_id].detach() + 1e-10))).sum(dim=-1).flatten()[
            sample_indices].mean()  # KL

        if 'sep' in self.args.agent:
            self.policy_optimizers[agent_id].zero_grad()
            policy_loss.backward()
            self.policy_optimizers[agent_id].step()
        else:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # Logging
            ess = (resampling_probs.sum() ** 2) / (resampling_probs ** 2).sum()
            self.logger.log_stat(f'effective_sample_size_{agent_id}', ess.detach().cpu(), t_env, to_sacred=False)
            self.logger.log_stat(f'ar_data_policy_loss_{agent_id}', ar_data_policy_loss.item(), t_env, to_sacred=False)

            self.logger.log_stat(f'policy_loss_{agent_id}', policy_loss.item(), t_env, to_sacred=False)
            self.logger.log_stat(f'nu_{agent_id}_loss', nu_loss.item(), t_env, to_sacred=False)

            if agent_id == 0:
                self.logger.log_stat(f'data_policy_loss_{agent_id}', data_policy_loss.item(), t_env, to_sacred=False)
                self.logger.log_stat(f'e_{agent_id}_loss', e_i_loss.item(), t_env, to_sacred=False)

            self.logger.log_stat(f'other_policy_ratio_mean_{agent_id}', other_policy_ratio_mean.item(), t_env, to_sacred=False)
            abs_states_sampled = states[:, :, -1].unsqueeze(-1).flatten()[sample_indices].sum() / mask_elems
            self.logger.log_stat(f'absorbing_states_sampled{agent_id}', abs_states_sampled.item(), t_env, to_sacred=False)
            self.logger.log_stat(f'absorbing_states_ratio{agent_id}', (states[:, :, -1].sum() / mask_elems).item(), t_env, to_sacred=False)

            if 'gfootball' not in self.args.env and self.args.env != "warehouse" and 'sc2' not in self.args.env:
                if self.args.env == "bridge":
                    optim_state_indices = th.where(rewards.cumsum(1)[:, -1] > -1.26)[0]
                    optim_mask = th.zeros_like(states)
                    optim_mask[optim_state_indices] = 1
                    sampled_mask = th.zeros_like(states).flatten()
                    sampled_mask[sample_indices] = 1
                    optim_sampled = sampled_mask * optim_mask.flatten()
                    self.logger.log_stat(f'optimal_states_sampled_{agent_id}', (optim_sampled.sum() / sample_indices.shape[0]).item(), t_env,
                                         to_sacred=False)
                    abs_states = th.where(states[:, :, -1] == 1)
                    optim_mask[abs_states[0], abs_states[1]] = 0
                    optim_sampled = sampled_mask * optim_mask.flatten()
                    self.logger.log_stat(f'optimal_states_sampled_nonabsorbing_{agent_id}',
                                         (optim_sampled.sum() / sample_indices.shape[0]).item(), t_env,
                                         to_sacred=False)

                for x in range(self.n_agents):
                    for y in range(self.n_actions):
                        self.logger.log_stat(f'policy_agent_{x}_action_{y}', policy_probs[:, 0, x, y].mean().item(),
                                             t_env, to_sacred=True)

    def cuda(self):
        self.mac.cuda()
        self.data_mac.cuda()
        self.ar_data_mac.cuda()
        for i in range(self.n_agents):
            self.nu_networks[i].cuda()
            self.e_networks[i].cuda()

    def save_models(self, path, training_state):
        self.mac.save_models(path)
        self.data_mac.save_models(path, prefix='data')
        self.ar_data_mac.save_models(path, prefix='ar_data')
        if 'sep' not in self.args.agent:
            th.save(self.policy_optimizer.state_dict(), f"{path}/policy_optimizer.th")
            th.save(self.ar_data_policy_optimizer.state_dict(), f"{path}/ar_data_policy_optimizer.th")
        for i in range(self.n_agents):
            th.save(self.nu_networks[i].state_dict(), f"{path}/nu_network_{i}.th")
            th.save(self.nu_optimizers[i].state_dict(), f"{path}/nu_optimizer_{i}.th")
            th.save(self.e_networks[i].state_dict(), f"{path}/e_network_{i}.th")
            th.save(self.e_optimizers[i].state_dict(), f"{path}/e_optimizer_{i}.th")
            if 'sep' in self.args.agent:
                th.save(self.policy_optimizers[i].state_dict(), f"{path}/policy_optimizer_{i}.th")
                th.save(self.ar_data_policy_optimizers[i].state_dict(), f"{path}/ar_data_policy_optimizer_{i}.th")
        th.save(self.data_policy_optimizer.state_dict(), f"{path}/data_policy_optimizer.th")
        th.save(training_state, f"{path}/training_state.th")

    def load_models(self, path):
        self.mac.load_models(path)
        self.data_mac.load_models(path, prefix='data')
        self.ar_data_mac.load_models(path, prefix='ar_data')
        if 'sep' not in self.args.agent:
            self.policy_optimizer.load_state_dict(th.load(f"{path}/policy_optimizer.th", map_location=lambda storage, loc: storage))
            self.ar_data_policy_optimizer.load_state_dict(th.load(f"{path}/ar_data_policy_optimizer.th", map_location=lambda storage, loc: storage))

        for i in range(self.n_agents):
            self.nu_networks[i].load_state_dict(th.load(f"{path}/nu_network_{i}.th", map_location=lambda storage, loc: storage))
            self.nu_optimizers[i].load_state_dict(th.load(f"{path}/nu_optimizer_{i}.th", map_location=lambda storage, loc: storage))
            self.e_networks[i].load_state_dict(th.load(f"{path}/e_network_{i}.th", map_location=lambda storage, loc: storage))
            self.e_optimizers[i].load_state_dict(th.load(f"{path}/e_optimizer_{i}.th", map_location=lambda storage, loc: storage))
            if 'sep' in self.args.agent:
                self.policy_optimizers[i].load_state_dict(th.load(f"{path}/policy_optimizer_{i}.th", map_location=lambda storage, loc: storage))
                self.ar_data_policy_optimizers[i].load_state_dict(th.load(f"{path}/ar_data_policy_optimizer_{i}.th", map_location=lambda storage, loc: storage))
        self.data_policy_optimizer.load_state_dict(
            th.load(f"{path}/data_policy_optimizer.th", map_location=lambda storage, loc: storage))
        training_state = th.load(f"{path}/training_state.th")
        return training_state
