from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from marl.env.multiagentenv import MultiAgentEnv

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging
import random


class mmdp_game1Env(MultiAgentEnv):
    """ICQ version of the MMDP (oroginally from QPLEX)
    """
    def __init__(
            self,
            n_agents=2,
            n_actions=100,
            reward_win=1000,
            episode_limit=50,
            obs_last_action=True,
            state_last_action=True,
            is_print=False,
            print_rew=False,
            print_steps=1000,
            seed=None,
            matrix_game=False,
            plus_reward_00=False,
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        # self.n_agents = 2
        self.n_agents = n_agents

        # Rewards args
        self.reward_win = reward_win

        # Other
        self._seed = seed

        # Actions
        self.n_states = 2
        self.n_actions = n_actions

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.p_step = 0
        self.rew_gather = []
        self.is_print_once = False

        self.episode_limit = episode_limit
        
        if self.n_agents == 2:
            self.R = np.zeros((self.n_states, self.n_actions, self.n_actions))
            # self.R[1, 0, 0] = self.reward_win

            if self.n_actions == 4:
                self.R[1, 0, 0] = self.reward_win
                self.R[1, -1, -1] = self.reward_win
                self.R[1, 0, 1] = -2
                self.R[1, 0, 2] = -2
                self.R[1, 0, -1] = -2
                self.R[1, 1, -1] = -2
                self.R[1, 2, -1] = -2

                self.R[1, 1, 0] = -1
                self.R[1, 2, 0] = -1
                self.R[1, -1, 0] = -1
                self.R[1, -1, 1] = -1
                self.R[1, -1, 2] = -1
            else:
                self.R[:] = 0
                if plus_reward_00:
                    self.R[1, 0, 0] = self.reward_win
                self.R[1, 0, -1] = self.reward_win
                self.R[1, -1, 0] = self.reward_win
                self.R[1, -1, -1] = -2

                self.R[0, :] = -2
                self.R[0, 0, 0] = self.reward_win
                self.R[0, -1, -1] = self.reward_win
            self.T = np.zeros((self.n_states, self.n_actions, self.n_actions)).astype('int32')
            # ICQ version
            # self.T[1, :, :] = 1
            # self.T[1, 0, 1] = 0
            # self.T[1, 1, 0] = 0
            # [
            #     [[0 0 0], [0 0 0], [0 0 0]],,
            #     [[1 0 1], [0 1 1], [1 1 1]]
            # ]


            # self.T[1, 0, 0] = 1
            self.T[1, 0, -1] = 1
            self.T[1, -1, 0] = 1
            self.T[0, -1, -1] = 1
            self.T[0, 0, 0] = 1
            if matrix_game:
                self.T[:] = 1
            # backup
            # if self.n_actions > 2:
            #     self.T[1, 1, 1] = 1
            #     self.T[1, 0, 2] = 1
            #     self.T[1, 2, 0] = 1

            # QPLEX version
            # self.T[1, 0, 0] = 1
            # self.T[1, 0, 1] = 1
            # self.T[1, 1, 0] = 1

        elif self.n_agents == 3:  # TODO
            self.R = np.zeros((self.n_states, self.n_actions, self.n_actions, self.n_actions))
            self.R[1, 0, 0] = self.reward_win

            self.T = np.zeros((self.n_states, self.n_actions, self.n_actions, self.n_actions)).astype('int32')
            self.T[1, 0, 0, 0] = 1 # transition if sum is less than 3/2
            self.T[1, 1, 0, 0] = 1
            self.T[1, 0, 1, 0] = 1
            self.T[1, 0, 0, 1] = 1
        
        self.state_now = 1 # np.random.binomial(1, 0.5)

        # Qatten
        self.unit_dim = self.n_states

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}
        
        if self.n_agents == 2:
            reward = self.R[self.state_now][actions[0]][actions[1]]
            self.state_now = self.T[self.state_now][actions[0]][actions[1]]
        elif self.n_agents == 3:
            reward = self.R[self.state_now][actions[0]][actions[1]][actions[2]]
            self.state_now = self.T[self.state_now][actions[0]][actions[1]][actions[2]]
        terminated = False
        info['battle_won'] = False

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return np.concatenate([np.eye(self.n_states)[self.state_now] for _ in range(self.n_agents)], axis=0)
        #return np.eye(self.n_states)[self.state_now]

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.n_agents * self.n_states

    def get_state(self):
        """Returns the global state."""
        return np.concatenate([np.eye(self.n_states)[self.state_now] for _ in range(self.n_agents)], axis=0)

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.n_agents * self.n_states

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        # self.state_now = np.random.binomial(1, 0.5)
        self.state_now = 1
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "unit_dim": self.unit_dim}
        return env_info

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game
        }
        return stats
