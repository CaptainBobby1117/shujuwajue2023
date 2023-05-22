import os
import time
import random
import logging
import torch
import numpy as np
from glob import glob
from logging import Logger
from tqdm.auto import tqdm
from torch_geometric.data import Batch


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


def get_logger(name, log_dir=None, log_fn='log.txt'):
    logger = logging.getLogger(name)                                                        #创建一个logger
    logger.setLevel(logging.DEBUG)                                                          #设置为DEBUG
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')     #定义输出格式

    stream_handler = logging.StreamHandler()                                                #创建一个handler用于输出到控制台
    stream_handler.setLevel(logging.DEBUG)                                                  #设置为DEBUG
    stream_handler.setFormatter(formatter)                                                  #设置输出格式
    logger.addHandler(stream_handler)                                                       #给logger添加handler

    if log_dir is not None:                                                         #若log_dir不为空，则额外加入一个写入文件用的handler
        file_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):   #返回值为root/prefix_时间戳_tag
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    

def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def repeat_data(data, num_repeat):
    datas = [data.clone() for i in range(num_repeat)]
    return Batch.from_data_list(datas)


def repeat_batch(batch, num_repeat):
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += datas.clone()
    return Batch.from_data_list(new_data)


def get_checkpoint_path(folder, it=None):
    if it is not None:
        return os.path.join(folder, '%d.pt' % it), it
    all_iters = list(map(lambda x: int(os.path.basename(x[:-3])), glob(os.path.join(folder, '*.pt'))))
    all_iters.sort()
    return os.path.join(folder, '%d.pt' % all_iters[-1]), all_iters[-1]
    