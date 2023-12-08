# AlberDICE: Addressing OOD Joint Actions in Offline MARL with Alternating DICE
Official PyTorch implementation for AlberDICE: 

https://arxiv.org/abs/2311.02194

## Note
 This codebase is based on  [PyMARL](https://github.com/oxwhirl/pymarl) which is open-sourced. 

## Setup
```shell
bash create_conda_env.sh
```


## Run an experiment 
```shell
conda activate alberdice
cd pymarl-master
```

### Matrix Game
```shell
# datasets: 02, 02_20, 00_02_20, random
python3 src/main.py --config=alberdice --env-config=mmdp_game_1 with agent=sep_mlp seed=0 load_buffer_id="random" buffer_size=16 batch_size=16 burn_in_period=16 test_greedy=False t_max=1100000
```
### Bridge

```shell
# # {load_buffer_id: buffer_size}: {"mix": 1000, "optimal": 500}
python3 src/main.py --config=alberdice --env-config=bridge with agent=sep_mlp env_args.start_adjacent=True seed=0 load_buffer_id="mix" \
buffer_size=1000 batch_size=128 burn_in_period=1000 test_greedy=False t_max=3100000
```

### Warehouse
```shell
# download dataset file
https://drive.google.com/drive/folders/1e7ttrZzCX2v8HsSMxjhy3Vrd7ZifYSOQ?usp=drive_link

# unzip dataset file
tar xvzf ./warehouse-tiny-2ag-expert.tar.gz -C ../buffer/
tar xvzf ./warehouse-tiny-4ag-expert.tar.gz -C ../buffer/
tar xvzf ./warehouse-tiny-6ag-expert.tar.gz -C ../buffer/
tar xvzf ./warehouse-small-2ag-expert.tar.gz -C ../buffer/
tar xvzf ./warehouse-small-4ag-expert.tar.gz -C ../buffer/
tar xvzf ./warehouse-small-6ag-expert.tar.gz -C ../buffer/

# {env_args.n_agents: 2, 4, 6 }
# {env_Args.scenario: rware-tiny, rware-small }
# {load_buffer_id: rware-tiny-2ag-easy-v4-expert, rware-small-2ag-easy-v4-expert, 
#                  rware-tiny-4ag-easy-v4-expert, rware-small-4ag-easy-v4-expert, 
#                  rware-tiny-6ag-easy-v4-expert, rware-small-6ag-easy-v4-expert }
python3 src/main.py --config=alberdice --env-config=warehouse with seed=0 env_args.scenario="rware-tiny"  \
env_args.n_agents=2 env_args.difficulty="easy" env_args.episode_limit=495 env_args.add_agent_id=False \
load_buffer_id=rware-tiny-2ag-easy-v4-expert buffer_size=1000 batch_size=32 burn_in_period=1000 is_batch_rl=True \
use_cuda=True test_greedy=False t_max=20100000 lr=0.0005 save_model=False standardize_reward=False \
standardize_obs=True test_nepisode=30 alpha_start=1 alpha_end=1 rnn_hidden_dim=128 agent=sep_transformer \
obs_agent_id=False nu_grad_penalty_coeff=0.1 nu_lr=0.0005 
```

Please cite our work as: 
```
@inproceedings{
matsunaga2023alberdice,
title={Alber{DICE}: Addressing Out-Of-Distribution Joint Actions in Offline Multi-Agent {RL} via Alternating Stationary Distribution Correction Estimation},
author={Daiki E. Matsunaga and Jongmin Lee and Jaeseok Yoon and Stefanos Leonardos and Pieter Abbeel and Kee-Eung Kim},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=LhVJdq4cZm}
}

```