from collections import defaultdict
import logging
import numpy as np
import wandb
import socket

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        self.use_wandb = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def setup_wandb(self):
        self.use_wandb = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0

        info = {'hist': {}}
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue

            # Google Football
            if "ball" in k or "active" in k or "roles" in k or "sticky" in k or "yellow_card" in k or "tired_factor" in k:
                continue
            if k == "right_team_mean" or k == "left_team_mean" or "direction" in k or "designated" in k:
                continue

            i += 1
            window = 1 if (k == "epsilon" or "test" in k) else 5
            import torch as th
            if isinstance(self.stats[k][-1][1], wandb.Histogram):
                item = np.mean([np.mean(x[1].bins) for x in self.stats[k][-window:]])
            elif isinstance(self.stats[k][-1][1], wandb.Image):
                item = 0
            elif "dumps" in k:
                item = 0
            else:
                try:
                    item = th.mean(th.tensor([x[1] for x in self.stats[k][-window:]]))
                except:
                    item = np.mean([x[1] for x in self.stats[k][-window:]])
            item_str = "{:.4f}".format(item)

            log_str += "{:<25}{:>8}".format(k + ":", item_str)
            log_str += "\n" if i % 4 == 0 else "\t"

            last_t, last_value = v[-1]

            # Google Football
            if "score_mean" in k:
                last_value = last_value[0]
            if isinstance(self.stats[k][-1][1], wandb.Histogram):
                info['hist'][k] = self.stats[k][-1][1]
            elif isinstance(self.stats[k][-1][1], wandb.Image):
                info[k] = self.stats[k][-1][1]
            elif "dumps" in k:
                for j, vid in enumerate(v[-1][-1]):
                    info[f"video_{j}"] = wandb.Video(vid, format='mp4')
            else:
                info[k] = item
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    hostname = socket.gethostname()
    if "condor" in hostname:
        logger.setLevel('ERROR')
        print("logger level set at ERROR")
    else:
        logger.setLevel('INFO')
        print("logger level set at INFO")
    return logger

