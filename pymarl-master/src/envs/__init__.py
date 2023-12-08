from functools import partial
from marl.env import MultiAgentEnv, mmdp_game1Env, BridgeEnvironment, WareHouseEnv

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "mmdp_game_1": partial(env_fn, env=mmdp_game1Env),
    "bridge": partial(env_fn, env=BridgeEnvironment, tabular_states=False),
    "bridge_tabular": partial(env_fn, env=BridgeEnvironment, tabular_states=True),
    "warehouse": partial(env_fn, env=WareHouseEnv)
}
