from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from torch.distributions import Categorical

import copy


class BasicMAC:
    def __init__(self, scheme, groups, args, data_policy=False, autoregressive=False):
        self.n_agents = args.n_agents
        self.data_policy = data_policy
        self.args = args
        self.agent_type = self.args.agent
        self.obs_agent_id = self.args.obs_agent_id

        self.autoregressive_data_policy = False
        if self.data_policy:
            if autoregressive:
                self.agent_type = "sep_mlp" if 'sep' in args.agent else 'mlp'
                self.autoregressive_data_policy = True
            else:
                if self.args.agent == "sep_mlp":
                    self.agent_type = "mlp"
                elif self.args.agent == "sep_transformer":
                    self.agent_type = "transformer"
                self.obs_agent_id = True

        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        if "sep" in self.agent_type:
            self.hidden_states = [None for _ in range(self.n_agents)]
        else:
            self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                                test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, agent_id=None):
        agent_inputs = self._build_inputs(ep_batch, t)  # [batch_size, n_agents, obs_dim]
        avail_actions = ep_batch["avail_actions"][:, t]
        actions_onehot = ep_batch["actions_onehot"][:, :t]
        actions_onehot = th.cat([actions_onehot, th.zeros(ep_batch.batch_size, 1, actions_onehot.shape[2],
                                                          actions_onehot.shape[-1], device=self.args.device)], dim=1)

        if "sep" in self.agent_type:
            agent_outs = []
            for i in range(self.args.n_agents):
                with th.set_grad_enabled(not test_mode):
                    if self.agent_type == "sep_transformer":
                        pi, self.hidden_states[i] = self.agent[i].forward(agent_inputs[:, :, i, :], actions_onehot[:, :, i, :],
                                                                          self.hidden_states[i])
                        pi = pi[:, -1] # output from forward is (a_0, a_1,...,a_t)
                    else:
                        pi, self.hidden_states[i] = self.agent[i].forward(agent_inputs[:, i, :], self.hidden_states[i])
                if agent_id is not None:
                    if agent_id != i:
                        agent_outs.append(pi.unsqueeze(dim=1).detach())
                        continue
                agent_outs.append(pi.unsqueeze(dim=1))
            agent_outs = th.cat(agent_outs, dim=1)
            agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
        else:
            with th.set_grad_enabled(not test_mode):
                if 'transformer' in self.agent_type:
                    batch_size, seq_len, n_agents, obs_dim = agent_inputs.shape
                    n_actions = actions_onehot.shape[-1]
                    obss_input = agent_inputs.permute(2, 0, 1, 3).reshape(n_agents * batch_size, seq_len, obs_dim)
                    actions_input = actions_onehot.permute(2, 0, 1, 3).reshape(n_agents * batch_size, seq_len, n_actions)

                    agent_outs, self.hidden_states = self.agent(obss_input, actions_input, self.hidden_states)  # [n_agents * batch_size, seq_len, n_actions]
                    agent_outs = agent_outs.view(n_agents, batch_size, seq_len, n_actions).permute(1, 2, 0, 3)  # [batch_size, seq_len, n_agents, n_actions]
                    agent_outs = agent_outs[:, -1, :, :].reshape(ep_batch.batch_size * self.n_agents, -1)
                else:
                    agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

            if th.where(agent_outs.isnan())[0].shape[0] > 0:
                pass
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def forward_policy(self, ep_batch, max_t, test_mode=False, agent_id=None):
        # batched forward for entire timesteps at once (assume MLP agent or Transformer)
        obss = ep_batch['obs'][:, :max_t]  # [batch_size, seq_len, n_agents, obs_dim]
        avail_actions = ep_batch["avail_actions"][:, :max_t]  # [batch_size, seq_len, n_agents, n_actions]
        actions_onehot = ep_batch["actions_onehot"][:, :max_t] # [batch_size, seq_len, n_agents, n_actions]

        if 'sep' in self.agent_type:
            agent_outs = []
            for i in range(self.n_agents):
                no_grad = (agent_id is not None and agent_id != i) or test_mode
                with th.set_grad_enabled(not no_grad):
                    if self.agent_type == "sep_transformer":
                        pi, _ = self.agent[i].forward(obss[:, :, i], actions_onehot[:, :, i])
                    else:
                        pi, _ = self.agent[i].forward(obss[:, :, i])

                agent_outs.append(pi.unsqueeze(2))
            agent_outs = th.cat(agent_outs, dim=2)  # [batch_size, seq_len, n_agents, n_actions]
        else:
            if self.obs_agent_id:
                # append agent-id to obss
                obss = th.cat([obss, th.eye(self.n_agents, device=obss.device).repeat(obss.shape[:2] + (1, 1))], dim=-1)  #   # [batch_size, seq_len, n_agents, state_dim + n_agents]
            if self.agent_type == 'transformer':
                batch_size, seq_len, n_agents, obs_dim = obss.shape
                n_actions = actions_onehot.shape[-1]
                obss_input = obss.permute(2, 0, 1, 3).reshape(n_agents * batch_size, seq_len, obs_dim)
                actions_input = actions_onehot.permute(2, 0, 1, 3).reshape(n_agents * batch_size, seq_len, n_actions)
                with th.set_grad_enabled(not test_mode):
                    agent_outs, self.hidden_states = self.agent(obss_input, actions_input, self.hidden_states)  # [n_agents * batch_size, seq_len, n_actions]
                agent_outs = agent_outs.view(n_agents, batch_size, seq_len, n_actions).permute(1, 2, 0, 3)  # [batch_size, seq_len, n_agents, n_actions]
            else:
                with th.set_grad_enabled(not test_mode):
                   agent_outs, self.hidden_states = self.agent(obss, self.hidden_states)  # [batch_size, seq_len, n_agents, n_actions]

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs[avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = avail_actions.sum(dim=-1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[avail_actions == 0] = 0.0

        return agent_outs

    def forward_ar_data_policy(self, ep_batch, max_t, agent_id, set_grad=True):
        # batched forward for entire timesteps at once (assume MLP agent)
        state = ep_batch['state'][:, :max_t]  # [batch_size, seq_len, state_dim]
        batch_size, seq_len, state_dim = state.shape

        avail_actions = ep_batch["avail_actions"][:, :max_t, th.arange(self.n_agents) != agent_id]  # [batch_size, seq_len, n_agents - 1, n_actions]
        avail_actions[avail_actions == 0] = -1e+10
        avail_actions[avail_actions == 1] = 0
        all_agents_actions = ep_batch["actions_onehot"][:, :max_t]
        n_actions = all_agents_actions.shape[-1]

        if 'sep' in self.args.agent:
            agent_outs = []
            state_action = th.zeros((batch_size, seq_len, state_dim + n_actions), device=state.device)
            state_action[:, :, :state_dim] = state
            state_action[:, :, state_dim:] = all_agents_actions[:, :, agent_id]

            other_actions = th.zeros((batch_size, seq_len, self.n_agents - 2, n_actions), device=state.device)  # (s, a_{-agent_id} ... )
            other_agent_ids = [x for x in range(self.n_agents) if x != agent_id]
            for idx, other_agent_id in enumerate(other_agent_ids):
                agent_id_input = th.zeros((batch_size, seq_len, self.n_agents), device=state.device)
                agent_id_input[:, :, other_agent_id] = 1
                agent_input = th.zeros([batch_size, seq_len, self.n_agents + state_action.shape[-1] + (self.n_agents - 2) * n_actions],
                                       device=state.device)
                agent_input[:, :, :self.n_agents] = agent_id_input
                agent_input[:, :, self.n_agents: self.n_agents + state_action.shape[-1]] = state_action
                agent_input[:, :, self.n_agents + state_action.shape[-1]:] = \
                    other_actions.reshape(batch_size, seq_len, (self.n_agents - 2) * n_actions)
                with th.set_grad_enabled(set_grad):
                    pi, _ = self.agent[agent_id].forward(agent_input)
                agent_outs.append(pi.unsqueeze(2))

                if idx < len(other_agent_ids) - 1:
                    other_actions[:, :, idx] = all_agents_actions[:, :, other_agent_id]
            agent_outs = th.cat(agent_outs, dim=2)  # [batch_size, seq_len, n_agents-1, n_actions]
        else:
            obss = state.unsqueeze(2).repeat((1, 1, self.n_agents - 1, 1))
            if self.args.obs_agent_id:
                # append agent-id to obss
                obss = th.cat([obss, th.eye(self.n_agents, device=obss.device)[agent_id].repeat((batch_size, seq_len, self.n_agents - 1, 1))], dim=-1)  # [batch_size, seq_len, n_agents-1, obs_dim]

            other_agent_ids = [x for x in range(self.n_agents) if x != agent_id]

            other_agent_actions = (
                all_agents_actions[:, :, other_agent_ids[:-1]].reshape(batch_size, seq_len, (self.n_agents - 2) * n_actions).unsqueeze(2).repeat((1, 1, self.n_agents-1, 1)) *
                th.repeat_interleave(th.tril(th.ones(self.n_agents - 1, self.n_agents - 2, device=obss.device), diagonal=-1), n_actions, dim=1)
            )
            obss = th.cat([
                th.eye(self.n_agents, device=obss.device)[other_agent_ids].repeat((batch_size, seq_len, 1, 1)),  # other_agent_id to predict
                obss,
                all_agents_actions[:, :, agent_id].unsqueeze(2).repeat(1, 1, self.n_agents - 1, 1),  # current agent's action is conditioned,  # other agent's actions
                other_agent_actions,
            ], dim=-1)
            with th.set_grad_enabled(set_grad):
                agent_outs, self.hidden_states = self.agent(obss, self.hidden_states)  # [batch_size, seq_len, n_agents-1, n_actions]

        # Softmax the agent outputs
        if self.args.mask_before_softmax:
            agent_outs += avail_actions
        agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)  # [batch_size, seq_len, n_agents-1, n_actions]

        return agent_outs

    def init_hidden(self, batch_size):
        if "sep" in self.agent_type:
            for i in range(self.n_agents):
                self.hidden_states[i] = self.agent[i].init_hidden().expand(batch_size, -1)  # bav
        else:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        if "sep" in self.agent_type:
            return [list(self.agent[i].parameters()) for i in range(self.n_agents)]
        else:
            return self.agent.parameters()

    def load_state(self, other_mac):
        if "sep" in self.agent_type:
            for i in range(self.n_agents):
                self.agent[i].load_state_dict(other_mac.agent[i].state_dict())
        else:
            self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        if "sep" in self.agent_type:
            for i in range(self.n_agents):
                self.agent[i].cuda()
        else:
            self.agent.cuda()

    def save_models(self, path, prefix=""):
        if prefix != "":
            prefix = prefix + "_"
        if "sep" in self.agent_type:
            for i in range(self.args.n_agents):
                th.save(self.agent[i].state_dict(), "{}/{}agent_{}.th".format(path, prefix, str(i)))
        else:
            th.save(self.agent.state_dict(), "{}/{}agent.th".format(path, prefix))

    def initialize_models(self, data_policy):
        for i in range(self.args.n_agents):
            self.agent[i].load_state_dict(data_policy[i].state_dict())

    def load_models(self, path, prefix=""):
        if prefix != "":
            prefix = prefix + "_"
        if "sep" in self.agent_type:
            for i in range(self.args.n_agents):
                self.agent[i].load_state_dict(th.load("{}/{}agent_{}.th".format(path, prefix, str(i)), map_location=lambda storage, loc: storage))
        else:
            self.agent.load_state_dict(th.load("{}/{}agent.th".format(path, prefix), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        agent_type = agent_REGISTRY[self.agent_type]
        if "sep" in self.agent_type:
            self.agent = []
            for i in range(self.args.n_agents):
                tmp_agent = agent_type(input_shape, self.args, i)
                self.agent.append(tmp_agent)
        else:
            self.agent = agent_type(input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        if 'transformer' in self.agent_type:
            inputs.append(batch["obs"][:, :t + 1])  # [batch_size, seq_len, n_agents, obs_dim]
        else:
            inputs.append(batch["obs"][:, t])  # b1av

        if self.args.obs_last_action:
            assert self.agent_type != "sep_transformer"
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])

        if self.agent_type == 'transformer':
            if self.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).repeat((bs, t + 1, 1, 1)))
            inputs = th.cat(inputs, dim=-1)
        elif "sep" not in self.agent_type:
            if self.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        else:
            inputs = th.cat(inputs, dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        if self.args.name == "bc_autoregressive":
            input_shape = scheme["state"]["vshape"] + (self.n_agents - 1) * scheme["actions_onehot"]["vshape"][0]
            # input_shape += self.n_agents
        elif self.data_policy:
            if self.autoregressive_data_policy:
                input_shape = scheme["state"]["vshape"] + (self.n_agents - 1) * scheme["actions_onehot"]["vshape"][0]
                input_shape += self.n_agents
            else:
                input_shape = scheme["obs"]["vshape"]
        else:
            input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.obs_agent_id and "sep" not in self.agent_type:
            input_shape += self.n_agents

        return input_shape
