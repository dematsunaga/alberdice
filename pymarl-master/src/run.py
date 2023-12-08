import datetime
import os
import pprint
import time
import threading
import json

import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import numpy as np
import copy as cp
import random
import glob
from tqdm import tqdm
import time


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    args.sacred_id = _run._id

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    datestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.is_batch_rl:
        buffer_id = args.load_buffer_id
    else:
        buffer_id = args.save_buffer_id
    if "optidice" in args.name or "alberdice" in args.name:
        name = args.name + f"-{args.alpha_start}-{args.alpha_end}"
    else: 
        name = args.name
    if args.env in ['mmdp_game_1', 'bridge', 'bridge_tabular'] or 'gfootball' in args.env:
        unique_token = f"{name}_{args.env}-{buffer_id}_{args.seed}_{datestamp}_sacred{_run._id}"
    elif args.env == "warehouse":
        unique_token = f"{name}_{args.env}_{args.env_args['scenario']}-{buffer_id}_{args.seed}_{datestamp}_sacred{_run._id}"
    else:
        unique_token = "{}_{}_{}-{}_{}_{}".format(name, args.env, args.env_args['map_name'], buffer_id, args.seed,
                                                  datestamp)
    args.unique_token = unique_token
    if args.use_tensorboard:
        # tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_logs_direc = os.path.join(args.local_results_path, "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    # logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def save_one_buffer(args, save_buffer, env_name, from_start=False):
    x_env_name = env_name
    if from_start:
        x_env_name += '_from_start/'

    if "mmdp" in x_env_name or "bridge" in x_env_name:
        path_name = args.buffer_path + x_env_name + '/buffer_' + str(args.save_buffer_id) + '/'
    else:
        path_name = args.buffer_path + x_env_name + '/buffer_' + str(args.env) + '_' + str(
            args.env_args['map_name']) + '_' + str(args.save_buffer_id) + '/'

    if os.path.exists(path_name):
        random_name = args.buffer_path + x_env_name + '/buffer_' + str(random.randint(10, 1000)) + '/'
        os.rename(path_name, random_name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    save_buffer.save(path_name)


def run_sequential(args, logger):
    # assert args.agent == 'sep_mlp'  # TODO: remove this

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.episode_limit = env_info["episode_limit"]
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["unit_dim"]

    # Default/Base scheme
    obs_shape = env_info["state_shape"] if args.obs_full_state else env_info["obs_shape"]
    args.obs_shape = obs_shape
    scheme = {
        "state": {"vshape": env_info["state_shape"] + int(args.absorbing_state)},
        "obs": {"vshape": obs_shape + int(args.absorbing_state) , "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    env_name = args.env
    if env_name == 'sc2':
        env_name += '/data/bk/clean_500/' + args.env_args['map_name'] + '_splited/good/'

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          args.burn_in_period,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device,
                          absorbing_state=args.absorbing_state,
                          standardize_obs=args.standardize_obs, standardize_reward=args.standardize_reward,
                          reward_weight=args.reward_weight, obs_full_state=args.obs_full_state)

    if args.is_save_buffer:
        save_buffer = ReplayBuffer(scheme, groups, args.save_buffer_size, env_info["episode_limit"] + 1,
                                   args.burn_in_period,
                                   preprocess=preprocess,
                                   device="cpu" if args.buffer_cpu_only else args.device)

    if args.is_batch_rl:
        assert (args.is_save_buffer == False)
        x_env_name = env_name
        if args.is_from_start:
            x_env_name += '_from_start/'

        path_name = args.buffer_path
        if "mmdp" in x_env_name or "bridge" in x_env_name:
            path_name += x_env_name
            path_name = path_name + '/buffer_'  + str(args.load_buffer_id) + '/'
        elif "gfootball" in x_env_name:
            if args.load_buffer_id == "medium-expert-1":
                medium_path_name = os.path.join(path_name, 'GoogleFootball', args.env_args['scenario_name'], 'medium_1_seeds/')
                expert_path_name = os.path.join(path_name, 'GoogleFootball', args.env_args['scenario_name'], 'expert_1_seeds/')
                assert os.path.exists(medium_path_name) == True, f"{medium_path_name} doesn't exist"
                assert os.path.exists(expert_path_name) == True, f"{expert_path_name} doesn't exist"
                buffer.load_medium_expert(medium_path_name, expert_path_name, args.buffer_size)
            elif args.load_buffer_id == "medium-expert-3":
                medium_path_name = os.path.join(path_name, 'GoogleFootball', args.env_args['scenario_name'], 'medium_3_seeds/')
                expert_path_name = os.path.join(path_name, 'GoogleFootball', args.env_args['scenario_name'], 'expert_3_seeds/')
                assert os.path.exists(medium_path_name) == True, f"{medium_path_name} doesn't exist"
                assert os.path.exists(expert_path_name) == True, f"{expert_path_name} doesn't exist"
                buffer.load_medium_expert(medium_path_name, expert_path_name, args.buffer_size)
            elif args.load_buffer_id == "random-expert-3":
                random_path_name = os.path.join(path_name, 'GoogleFootball', args.env_args['scenario_name'],
                                                'random/')
                expert_path_name = os.path.join(path_name, 'GoogleFootball', args.env_args['scenario_name'],
                                                'expert_3_seeds/')
                assert os.path.exists(random_path_name) == True, f"{random_path_name} doesn't exist"
                assert os.path.exists(expert_path_name) == True, f"{expert_path_name} doesn't exist"
                buffer.load_medium_expert(random_path_name, expert_path_name, args.buffer_size)
            else:
                path_name = os.path.join(path_name, 'GoogleFootball', args.env_args['scenario_name'], f'{args.load_buffer_id}/')
                buffer.load(path_name, args.buffer_size)
        elif "warehouse" in x_env_name:
            path_name = path_name + str(args.env) + '/' + str(
                args.env_args['scenario']) + '/' + str(args.load_buffer_id) + '/'
        elif "sc2" in env_name:
            path_name += env_name
            buffer.load(path_name, args.buffer_size)
        else:
            path_name += x_env_name
            path_name = path_name + '/buffer_' + str(args.env) + '_' + str(
                args.env_args['map_name']) + '_' + str(args.load_buffer_id) + '/'
            buffer.load(path_name, args.buffer_size)
        # path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.load_buffer_id) + '/'
        if "warehouse" in x_env_name or "mmdp" in x_env_name or "bridge" in x_env_name:
            assert os.path.exists(path_name), f"{path_name} doesn't exist"
            buffer.load(path_name, args.buffer_size)

        args.dataset_return_per_ep = buffer.data.transition_data["reward"].sum(1).mean().item()
        if buffer.reward_configs is not None:
            args.good_eps_return_mean = buffer.reward_configs["good_eps_return_mean"]
            args.bad_eps_return_mean = buffer.reward_configs["bad_eps_return_mean"]
            args.num_good_episodes = buffer.reward_configs["num_good_episodes"]
            args.num_bad_episodes = buffer.reward_configs["num_bad_episodes"]
            args.good_episodes_ratio = buffer.reward_configs["good_episodes_ratio"]
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    if args.learner == 'alberdice_learner':
        data_mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, data_policy=True)
        ar_data_mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, data_policy=True, autoregressive=True)
        learner = le_REGISTRY[args.learner](mac, data_mac, ar_data_mac, buffer.scheme, logger, args)
    else:
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        mac.cuda()
        if args.learner == 'alberdice_learner':
            ar_data_mac.cuda()
            data_mac.cuda()
        learner.cuda()

    training_state = {
        'episode': 0,
        'last_test_T': -args.test_interval - 1,
        'last_test_T_log': -args.test_interval - 1,
        'last_log_T': 0, # -args.log_interval - 1,
        'model_save_time': 0,
        'last_buffer_save_T': 0,
        'wandb_run_id': None
    }

    if args.checkpoint_path != "":
        os.makedirs(args.checkpoint_path, exist_ok=True)

        # checkpoint_dirs = sorted(glob.glob(os.path.join(args.checkpoint_path, '*')), key=os.path.getmtime)
        timesteps = list(sorted([int(x) for x in os.listdir(args.checkpoint_path) if 'tmp' not in x]))

        if args.load_step != 0 and args.load_step not in timesteps:
            logger.console_logger.info(f"{os.path.join(args.checkpoint_path, str(args.load_step))} doesn't exists!")
            return

        if len(timesteps) > 0:
            if args.load_step == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

            model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

            logger.console_logger.info("Loading model from {}".format(model_path))
            training_state = learner.load_models(model_path)
            runner.t_env = timestep_to_load

            if args.evaluate or args.save_replay:
                evaluate_sequential(args, runner)
                return

    start_time = time.time()
    last_time = start_time
    if not args.no_train:
        logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    if args.env in ['mmdp_game_1']:
        training_state['last_demo_T'] = -args.demo_interval - 1

    def sample_batch():
        episode_sample = buffer.sample(args.batch_size)

        # Truncate batch to only filled timesteps
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]

        if episode_sample.device != args.device:
            episode_sample.to(args.device)
        return episode_sample

    pbar = tqdm(desc=f'{args.name}', ncols=80, total=args.t_max)
    pbar.update(runner.t_env)
    while runner.t_env <= args.t_max:
        if not args.is_batch_rl:
            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False, no_train=args.no_train)
            buffer.insert_episode_batch(episode_batch)

            if args.is_save_buffer:
                save_buffer.insert_episode_batch(episode_batch)
                if save_buffer.episodes_in_buffer == save_buffer.buffer_size:
                    if save_buffer.is_from_start:
                        save_buffer.is_from_start = False  # TODO
                        save_one_buffer(args, save_buffer, env_name, from_start=True)
                    else:
                        if args.save_buffer_during_train and (runner.t_env - training_state['last_buffer_save_T']) / args.save_buffer_interval >= 1.0 :
                            save_one_buffer(args, save_buffer, env_name, from_start=False)
                if save_buffer.buffer_index % args.save_buffer_interval == 0:
                    print('current episodes_in_buffer: ', save_buffer.episodes_in_buffer)

        if not args.no_train:
            for circle in range(args.num_circle):
                if buffer.can_sample(args.batch_size):
                    episode_sample = sample_batch()

                    if args.is_batch_rl:
                        if args.learner == 'alberdice_learner':
                            t_env_increment =  int(args.num_iter_inner_loop * th.sum(episode_sample['filled']).cpu().clone().detach().numpy()) // args.batch_size
                        else:
                            t_env_increment =  int(th.sum(episode_sample['filled']).cpu().clone().detach().numpy()) // args.batch_size
                        runner.t_env += t_env_increment
                        pbar.update(t_env_increment)

                    if args.learner == 'alberdice_learner':
                        learner.train(sample_batch, runner.t_env, training_state['episode'])
                    else:
                        learner.train(episode_sample, runner.t_env, training_state['episode'])

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - training_state['last_test_T']) / args.test_interval >= 1.0 :
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, training_state['last_test_T'], runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            training_state['last_test_T'] = runner.t_env
            training_state['last_test_T_log'] += args.test_interval
            if args.use_vec_eval:
                runner.run_vecenv(
                    test_mode=True, state_means=buffer.state_means, last_test_T=training_state['last_test_T_log'],
                    state_stds=buffer.state_stds, obs_means=buffer.obs_means, obs_stds=buffer.obs_stds)
            else:
                start_time = time.time()
                for _ in range(n_test_runs):
                    if args.is_batch_rl and args.standardize_obs:
                        runner.run(test_mode=True, state_means=buffer.state_means, last_test_T=training_state['last_test_T_log'],
                                state_stds=buffer.state_stds, obs_means=buffer.obs_means, obs_stds=buffer.obs_stds)
                    else:
                        runner.run(test_mode=True, last_test_T=training_state['last_test_T_log'])
                print(f'Evaluation elapsed time: {time.time() - start_time}')

        if args.save_model and (runner.t_env - training_state['model_save_time'] >= args.save_model_interval or training_state['model_save_time'] == 0):
            delete_prev_t = False
            if args.replace_prev_checkpoint and os.path.exists(args.checkpoint_path):
                t_list = list(sorted([int(x) for x in os.listdir(args.checkpoint_path)]))
                if len(t_list) > 0:
                    prev_t = t_list[0]
                    delete_prev_t = True
            training_state['model_save_time'] = runner.t_env
            save_path = os.path.join(args.checkpoint_path, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path + '.tmp', exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path + '.tmp', training_state)
            os.rename(save_path + '.tmp', save_path)
            if delete_prev_t:
                for file in os.listdir(os.path.join(args.checkpoint_path, str(prev_t))):
                    try:
                        os.remove(os.path.join(args.checkpoint_path, str(prev_t), file))
                    except OSError as e:
                        print(e)
                        print(f"Could not delete {os.path.join(args.checkpoint_path, str(prev_t), file)}")
                try:
                    os.rmdir(os.path.join(args.checkpoint_path, str(prev_t)))
                except OSError as e:
                    print(e)
                    print(f"Could not delete {os.path.join(args.checkpoint_path, str(prev_t))}")

        training_state['episode'] += args.batch_size_run * args.num_circle

        if (runner.t_env - training_state['last_log_T']) >= args.log_interval:
            logger.log_stat("episode", training_state['episode'], runner.t_env)
            logger.print_recent_stats()

            training_state['last_log_T'] = runner.t_env

    # if args.is_save_buffer and save_buffer.is_from_start:
    #     save_buffer.is_from_start = False
    #     save_one_buffer(args, save_buffer, env_name, from_start=True)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
