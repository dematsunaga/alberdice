from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from tqdm import tqdm
import time

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # for Gfootball since rewards \neq goals
        self.test_num_goals = 0

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        if 'gfootball' in self.args.env or self.args.env == "warehouse" or "sc2" in self.args.env:
            obs, state, env_info = self.env.reset()
            self.t = 0
            return obs, state, env_info
        else:
            self.env.reset()
            self.t = 0

    def run(self, test_mode=False, no_train=False, state_means=None, state_stds=None, obs_means=None, obs_stds=None,
            last_test_T=0):
        if 'gfootball' in self.args.env or self.args.env == "warehouse" or "sc2" in self.args.env:
            obs, state, env_info = self.reset()
            if self.args.obs_full_state:
                obs = [state for _ in range(self.args.n_agents)]
        else:
            self.reset()

        terminated = False
        episode_return = 0
        if "rnn" in self.args.agent:
            self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            if 'gfootball' in self.args.env or self.args.env == "warehouse" or "sc2" in self.args.env:
                pre_transition_data = {
                    "state": [state],
                    "avail_actions": [env_info["avail_actions"]],
                    "obs": [obs]
                }
            else:
                pre_transition_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()]
                }
            if self.args.absorbing_state:
                pre_transition_data["state"][0] = np.pad(pre_transition_data["state"][0], (0, 1))
                pre_transition_data["obs"][0] = np.pad(pre_transition_data["obs"][0], ((0, 0), (0, 1)))

            if self.args.standardize_obs:
                assert state_means is not None and state_stds is not None
                assert obs_means is not None and obs_stds is not None
                pre_transition_data["state"][0] = (pre_transition_data["state"][0] - state_means[0]) / \
                                                  (state_stds[0] + 1e-10)
                pre_transition_data["obs"][0] = (pre_transition_data["obs"][0] - obs_means) / (obs_stds + 1e-10)

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=(test_mode or no_train))

            if "gfootball" in self.args.env or self.args.env == "warehouse" or "sc2" in self.args.env:
                obs, reward, state, terminated, env_info = self.env.step(actions[0], self.t)
                if self.args.obs_full_state:
                    obs = [state for _ in range(self.args.n_agents)]

                episode_return += reward

                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated, )],
                }
            else:
                reward, terminated, env_info = self.env.step(actions[0])
                episode_return += reward

                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
        if 'gfootball' not in self.args.env and self.args.env != "warehouse" and "sc2" not in self.args.env:
            last_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
        else:
            last_data = {
                "state": [state],
                "avail_actions": [env_info["avail_actions"]],
                "obs": [obs]
            }
            env_info.pop("avail_actions", None)

        if self.args.absorbing_state:
            last_data["state"][0] = np.pad(last_data["state"][0], (0, 1))
            last_data["obs"][0] = np.pad(last_data["obs"][0], ((0, 0), (0, 1)))

        if self.args.standardize_obs:
            assert state_means is not None and state_stds is not None
            assert obs_means is not None and obs_stds is not None
            last_data["state"][0] = (last_data["state"][0] - state_means[0]) / (state_stds[0] + 1e-10)
            last_data["obs"][0] = (last_data["obs"][0] - obs_means) / (obs_stds + 1e-10)

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=(test_mode or no_train))

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info) if k != 'dumps'})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix, last_test_T=last_test_T)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            # if hasattr(self.mac.action_selector, "epsilon"):
            #     self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix, last_test_T=None):
        if last_test_T is not None:
            t_env = last_test_T
        else:
            t_env = self.t_env
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), t_env)
        self.logger.log_stat(prefix + "return_median", np.median(returns), t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), t_env)

        returns.clear()

        for k, v in stats.items():
            if k == "dumps":
                self.logger.log_stat(prefix + k, v, t_env)
            elif k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v/stats["n_episodes"], t_env)
        stats.clear()
