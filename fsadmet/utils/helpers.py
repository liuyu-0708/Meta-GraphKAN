
import torch
import torch.nn as nn
import random
import numpy as np
from torch.backends import cudnn


def get_device(cfg):
    device = torch.device("cuda:{}".format(cfg.train.device_ids[0]
                                           ) if torch.cuda.is_available()
                          and len(cfg.train.device_ids) > 0 else "cpu")
    return device


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel)


def fix_random_seed(random_seed, cuda_deterministic=True):
    # fix random seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
