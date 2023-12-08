import torch as th
import numpy as np
from types import SimpleNamespace as SN
import json
import torch.nn.functional as F
from tqdm import tqdm
import os

class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu",
                 absorbing_state=False,
                 standardize_obs=False,
                 standardize_reward=False,
                 reward_weight=1,
                 obs_full_state=False):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device
        self.absorbing_state = absorbing_state
        self.standardize_obs = standardize_obs
        self.standardize_reward = standardize_reward
        self.reward_weight = reward_weight
        self.obs_full_state = obs_full_state
        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, burn_in_period, preprocess=None, device="cpu",
                 absorbing_state=False, standardize_obs=False, standardize_reward=False, reward_weight=1,
                 obs_full_state=False):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess,
                                           device=device, absorbing_state=absorbing_state, standardize_obs=standardize_obs,
                                           standardize_reward=standardize_reward, reward_weight=reward_weight,
                                           obs_full_state=obs_full_state)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.is_from_start = True
        self.burn_in_period = burn_in_period

        self.reward_configs = None # dict for storing dataset statistics

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= self.burn_in_period

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())

    def load_numpy_data(self, path_name, buffer_size, medium_expert=False):
        num_agents = self.data.transition_data["obs"].shape[2]
        if type(path_name) == str:
            path_name = [path_name]
        # Load dataset
        for key, item in self.data.transition_data.items():
            aggregated_value = []
            for path in path_name:
                file_name = path + key + '.npy'
                if key == 'obs' and (self.obs_full_state or not os.path.exists(file_name)):
                    print(f"{file_name} ignored or not found. Using state.npy as local observations..")
                    file_name = path + 'state.npy'
                    data = th.from_numpy(np.load(file_name))  # [num_episodes, num_timesteps, state_dim]
                    data = data.unsqueeze(2).expand(-1, -1, num_agents,-1)  # [num_episodes, num_timesteps, num_agents, state_dim]
                else:
                    data = th.from_numpy(np.load(file_name))

                if medium_expert:
                    data = data[:buffer_size//2, :]

                if self.device == 'gpu':
                    data = data.gpu()

                if self.absorbing_state and key in ['state', 'obs']:
                    # add one dim to indicate whether it is an absorbing state or not.
                    data = th.cat([data, th.zeros(data.shape[:-1] + (1,))], dim=-1)
                aggregated_value.append(data)

            aggregated_value = th.cat(aggregated_value)

            # assume that some datssets can exceed the max # of timesteps
            self.data.transition_data[key] = aggregated_value[:buffer_size, :self.data.transition_data[key].shape[1]]

        assert (self.data.transition_data["actions"].squeeze() - self.data.transition_data["actions_onehot"].argmax(-1).squeeze()).sum() == 0

        print("previous reward max: ", self.data.transition_data["reward"].max())
        print("previous reward mean: ", self.data.transition_data["reward"].mean())
        print("previous reward min: ", self.data.transition_data["reward"].min())
        print('terminated timestep mean ', th.where(self.data.transition_data["terminated"] == 1)[1].float().mean())
        terminals_1 = th.zeros_like(self.data.transition_data["terminated"])
        terminals_1[:, 1:] = self.data.transition_data["terminated"][:, :-1]
        mask_1 = terminals_1.cumsum(dim=1)
        if self.standardize_reward:
            # mean/std taken over the states including up to the terminal state
            r = self.data.transition_data["reward"]
            r_mean = ((1 - mask_1) * r).sum() / (1 - mask_1).sum()
            r_std = th.sqrt((((r - r_mean)**2) * (1 - mask_1)).sum() / (1 - mask_1).sum() )
            self.data.transition_data["reward"] = (r - r_mean) / (r_std + 1e-10) * self.reward_weight
            self.data.transition_data["reward"] *= (1 - mask_1)
            print("after reward max: ", self.data.transition_data["reward"].max())
            print("after reward mean: ", self.data.transition_data["reward"].mean())
            print("after reward min: ", self.data.transition_data["reward"].min())

        terminals_2 = th.zeros_like(self.data.transition_data["terminated"])
        terminals_2[:, 2:] = self.data.transition_data["terminated"][:, :-2]
        mask_2 = terminals_2.cumsum(dim=1).expand(-1, -1, self.data.transition_data["state"].shape[-1])
        mask_2_obs = terminals_2.cumsum(dim=1).unsqueeze(-1).expand(-1, -1, num_agents, self.data.transition_data["obs"].shape[-1])
        if self.standardize_obs:
            # mean/std taken over the states including up to the 1st state after the terminal state
            s = self.data.transition_data["state"]

            s_means = (s * (1 - mask_2)).sum(dim=0).sum(dim=0) / (1 - mask_2).sum(dim=0).sum(dim=0)
            s_stds = th.sqrt((((s - s_means)**2) * (1 - mask_2)).sum(dim=0).sum(dim=0) / (1 - mask_2).sum(dim=0).sum(dim=0))
            if self.absorbing_state:
                s_means[-1] = 0
                s_stds[-1] = 1 - 1e-10

            self.data.transition_data["state"] = (s - s_means) / (s_stds + 1e-10)
            self.data.transition_data["state"] *= (1 - mask_2)

            # for evaluation later
            self.state_means = s_means.expand(self.data.transition_data["obs"].shape[2], -1).numpy()
            self.state_stds = s_stds.expand(self.data.transition_data["obs"].shape[2], -1).numpy()

            # repeat process for obs
            o = self.data.transition_data["obs"]
            o_means = (o * (1 - mask_2_obs)).sum(dim=0).sum(dim=0) / (1 - mask_2_obs).sum(dim=0).sum(dim=0)
            o_stds = th.sqrt(
                (((o - o_means) ** 2) * (1 - mask_2_obs)).sum(dim=0).sum(dim=0) / (1 - mask_2_obs).sum(dim=0).sum(dim=0))
            if self.absorbing_state:
                o_means[-1] = 0
                o_stds[-1] = 1 - 1e-10
            self.data.transition_data["obs"] = (o - o_means) / (o_stds + 1e-10)
            self.data.transition_data["obs"] *= (1 - mask_2_obs)

            # for evaluation later
            self.obs_means = o_means.expand(self.data.transition_data["obs"].shape[2], -1).numpy()
            self.obs_stds = o_stds.expand(self.data.transition_data["obs"].shape[2], -1).numpy()

        # To avoid: RuntimeError: expected scalar type Double but found Float
        self.data.transition_data["state"] = self.data.transition_data["state"].float()
        self.data.transition_data["obs"] = self.data.transition_data["obs"].float()
        self.data.transition_data['actions_onehot'] = self.data.transition_data['actions_onehot'].float()
        self.data.transition_data['actions'] = self.data.transition_data['actions'].float()
        self.data.transition_data['avail_actions'] = self.data.transition_data['avail_actions'].float()
        # Data preprocessing...
        if self.absorbing_state:
            terminal_states = th.where(mask_2 == 1)
            self.data.transition_data["state"][terminal_states[0], terminal_states[1], -1] = 1
            self.data.transition_data["obs"][terminal_states[0], terminal_states[1], :, -1] = 1

            # 'state', 'obs', 'actions', 'avail_actions', 'reward', 'terminated', 'actions_onehot', 'filled'
            max_timestep = self.data.transition_data['state'].shape[1]
            for i in tqdm(range(buffer_size), ncols=100, desc='absorbing_state'):
                terminated_timestep = -1
                terminated_r = 0
                terminated = False
                t = 0
                while t < max_timestep:
                    if self.data.transition_data['terminated'][i, t, 0] == 1:
                        assert not terminated
                        self.data.transition_data['terminated'][i, t, 0] = 0
                        terminated = True
                        terminated_timestep = t
                        terminated_r = self.data.transition_data['reward'][i, terminated_timestep, 0]

                        # cases where (terminated_timestep + 1)th timestep has no actions
                        if t < max_timestep - 1 and self.data.transition_data['actions_onehot'][i, t + 1, :, :].sum() == 0:
                            rand_action_idx = np.random.randint(0, terminated_timestep + 1)
                            self.data.transition_data['actions'][i, t + 1, :, :] = self.data.transition_data['actions'][i, rand_action_idx, :, :]
                            self.data.transition_data['actions_onehot'][i, t + 1, :, :] = self.data.transition_data['actions_onehot'][i, rand_action_idx, :, :]
                            self.data.transition_data['avail_actions'][i, t + 1, :, :] = self.data.transition_data['avail_actions'][i, rand_action_idx, :, :]
                        t += 2
                        continue
                    if terminated:
                        # comment out for now since the dataset may contain terminated states with non-zero obs
                        # assert (self.data.transition_data['state'][i, t] == 0).all()
                        # assert (self.data.transition_data['obs'][i, t] == 0).all()
                        # assert (self.data.transition_data['actions'][i, t] == 0).all()
                        # assert (self.data.transition_data['actions_onehot'][i, t] == 0).all()
                        # assert (self.data.transition_data['avail_actions'][i, t] == 0).all()
                        self.data.transition_data['state'][i, t, :-1] = 0  # aborsbing state
                        self.data.transition_data['obs'][i, t, :-1] = 0  # aborsbing state

                        self.data.transition_data['state'][i, t, -1] = 1  # absorbing state (last dimension)
                        self.data.transition_data['obs'][i, t, :, -1] = 1  # absorbing state (last dimension)

                        self.data.transition_data['filled'][i, t, 0] = 1
                        self.data.transition_data['reward'][i, t, 0] = 0
                        rand_action_idx = np.random.randint(0, terminated_timestep + 1)
                        self.data.transition_data['actions'][i, t, :, :] = self.data.transition_data['actions'][i, rand_action_idx, :, :]
                        self.data.transition_data['actions_onehot'][i, t, :, :] = self.data.transition_data['actions_onehot'][i, rand_action_idx, :, :]
                        self.data.transition_data['avail_actions'][i, t, :, :] = self.data.transition_data['avail_actions'][i, rand_action_idx, :, :]
                    t += 1
            assert th.where(self.data.transition_data['filled'] != 1)[0].shape[0] == 0 # all data filled
            assert self.data.transition_data["terminated"].sum() == 0 # no terminated flags
            assert th.where(self.data.transition_data['actions_onehot'].sum(-1) == 0)[0].shape[0] == 0 # actions filled
            assert (self.data.transition_data["actions"].squeeze() - self.data.transition_data["actions_onehot"].argmax(
                -1).squeeze()).sum() == 0
            assert th.where(self.data.transition_data['avail_actions'].sum(-1) == 0)[0].shape[0] == 0 # avail actions filled

        for key, item in self.data.episode_data.items():
            file_name = path_name + key + '.npy'
            data = th.from_numpy(np.load(file_name))
            if self.device == 'gpu':
                data = data.gpu()
            self.data.episode_data[key] = data[:buffer_size]

    def load(self, path_name, buffer_size):
        print('start loading buffer!')
        file_name = path_name + 'meta.json'
        with open(file_name) as fd:
            meta = json.load(fd)
        self.load_numpy_data(path_name, meta['episodes_in_buffer'])
        self.buffer_index = meta['buffer_index']
        # self.episodes_in_buffer = meta['episodes_in_buffer']
        # self.buffer_size = meta['buffer_size']
        self.episodes_in_buffer = buffer_size
        self.buffer_size = buffer_size
        print('episodes_in_buffer: ', self.episodes_in_buffer)
        print('finish loading buffer!')

    def load_medium_expert(self, path_name_med, path_name_expert, buffer_size):
        print('start loading buffer!')

        file_name_medium = path_name_med + 'meta.json'
        with open(file_name_medium) as fd:
            meta_medium = json.load(fd)
        file_name_expert = path_name_expert + 'meta.json'
        with open(file_name_expert) as fd:
            meta_expert = json.load(fd)
        if meta_medium['episodes_in_buffer'] + meta_expert['episodes_in_buffer'] > buffer_size:
            self.load_numpy_data((path_name_med, path_name_expert), buffer_size, medium_expert=True)
        else:
            # assert self.buffer_size == meta_medium['episodes_in_buffer'] + meta_expert['episodes_in_buffer']
            self.load_numpy_data((path_name_med, path_name_expert),
                                 meta_medium['episodes_in_buffer'] + meta_expert['episodes_in_buffer'], medium_expert=True)
        self.buffer_index = meta_medium['buffer_index']
        # self.episodes_in_buffer = meta['episodes_in_buffer']
        # self.buffer_size = meta['buffer_size']
        self.episodes_in_buffer = buffer_size
        self.buffer_size = buffer_size
        print('episodes_in_buffer: ', self.episodes_in_buffer)
        print('finish loading buffer!')

    def save_numpy_data(self, path_name):
        for key, item in self.data.transition_data.items():
            file_name = path_name + key + '.npy'
            data = item.cpu().clone().detach().numpy()
            np.save(file_name, data)
        for key, item in self.data.episode_data.items():
            file_name = path_name + key + '.npy'
            data = item.cpu().clone().detach().numpy()
            np.save(file_name, data)

    def save(self, path_name):
        print('start saving buffer!')
        print('episodes_in_buffer: ', self.episodes_in_buffer)
        self.save_numpy_data(path_name)
        file_name = path_name + 'meta.json'
        meta = {'buffer_index': self.buffer_index,
                'episodes_in_buffer': self.episodes_in_buffer,
                'buffer_size': self.buffer_size}
        with open(file_name, 'w') as fp:
            json.dump(meta, fp)
        print('finish saving buffer!')
