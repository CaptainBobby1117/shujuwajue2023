import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader

from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.transforms import *
from utils.misc import *
from utils.common import get_optimizer, get_scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/qm9_default.yml')     # 选择 config 文件
    parser.add_argument('--device', type=str, default='cuda')                   
    parser.add_argument('--resume_iter', type=int, default=None)                
    parser.add_argument('--logdir', type=str, default='./logs')                 
    args = parser.parse_args()
    os.chdir('./myc' )
    resume = os.path.isdir(args.config)                                             # 判断设置的 config 是否为文件夹
    if resume:                                                                      
        config_path = glob(os.path.join(args.config, '*.yml'))[0]                   # 如果是文件夹则将路径补完
        resume_from = args.config                                                   # resume_from 即为 config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))                                        # 读取配置文件
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]      
                                                                                    # config_name 为配置文件的文件名（如qm9_default）
    seed_all(config.train.seed)                                                     # 设置随机数种子？

    
    # Logging
    if resume:                                                                      # 如果 config 是文件夹
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')    # 
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)                  # 给出一个dir，例如：'./logs/qm9_default_2022_11_07__11_55_53'
        shutil.copytree('/home/ycmeng/myc/models', os.path.join(log_dir, 'models')) # 复制文件夹，将 models 里面的复制到 log_dir/model 里面去
    ckpt_dir = os.path.join(log_dir, 'checkpoints')                                 # 从 log_dir 创建 checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)                                            # 从 log_dir 创建 checkpoint
    logger = get_logger('train', log_dir)                                           # 创建一个而用于输出的 logger 
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)                         # 定义一个 SummaryWriter
    logger.info(args)                                                               # 输出 args 数据
    logger.info(config)                                                             # 输出 config 数据
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))  # 把 config_path 中的文件复制到 log_dir 中

    logger.info(os.getcwd().replace('\\','/'))
    # Datasets and loaders
    logger.info('Loading datasets...')
    transforms = CountNodesPerGraph()                                               # 用于返回数据中点的个数
    train_set = ConformationDataset(config.dataset.train, transform=transforms)     # 训练集构造，transform里面的部分用于返回数据中点的个数
    val_set = ConformationDataset(config.dataset.val, transform=transforms)         # 验证集构造，transform里面的部分用于返回数据中点的个数
    train_iterator = inf_iterator(DataLoader(train_set, config.train.batch_size, shuffle=True))
                                                                                    # 训练集迭代器构造，且每个epoch时都shuffle其中的batch
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)        # 验证集读取器构造，不需要shuffle其中的batch

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)                                 # 构造 model

    # Optimizer
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)
    start_iter = 1

####################################################################
    resume = True                   #自己改的部分（默认文件夹为resume有些诡异）
####################################################################

    # Resume from checkpoint
    if resume:                                                                      # 如果继续之前的运行（比较奇怪的是resume是按照config是否为文件夹判断的）
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)

        logger.info('Resuming from: %s' % ckpt_path)            
        logger.info('Iteration: %d' % start_iter)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer_global.load_state_dict(ckpt['optimizer_global'])
        optimizer_local.load_state_dict(ckpt['optimizer_local'])
        scheduler_global.load_state_dict(ckpt['scheduler_global'])
        scheduler_local.load_state_dict(ckpt['scheduler_local'])

    def train(it):
        model.train()
        optimizer_global.zero_grad()
        optimizer_local.zero_grad()
        batch = next(train_iterator).to(args.device)
        loss, loss_global, loss_local = model.get_loss(
            atom_type=batch.atom_type,
            pos=batch.pos,
            bond_index=batch.edge_index,
            bond_type=batch.edge_type,
            batch=batch.batch,
            num_nodes_per_graph=batch.num_nodes_per_graph,
            num_graphs=batch.num_graphs,
            anneal_power=config.train.anneal_power,
            return_unreduced_loss=True
        )
        loss = loss.mean()      # 求平均数
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)    # 梯度裁剪，将过大的梯度减小到阈值
        optimizer_global.step()
        optimizer_local.step()

        logger.info('[Train] Iter %05d | Loss %.2f | Loss(Global) %.2f | Loss(Local) %.2f | Grad %.2f | LR(Global) %.6f | LR(Local) %.6f' % (
            it, loss.item(), loss_global.mean().item(), loss_local.mean().item(), orig_grad_norm, optimizer_global.param_groups[0]['lr'], optimizer_local.param_groups[0]['lr'],
        ))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/loss_global', loss_global.mean(), it)
        writer.add_scalar('train/loss_local', loss_local.mean(), it)
        writer.add_scalar('train/lr_global', optimizer_global.param_groups[0]['lr'], it)
        writer.add_scalar('train/lr_local', optimizer_local.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad_norm', orig_grad_norm, it)
        writer.flush()

    def validate(it):
        sum_loss, sum_n = 0, 0
        sum_loss_global, sum_n_global = 0, 0
        sum_loss_local, sum_n_local = 0, 0
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
                batch = batch.to(args.device)
                loss, loss_global, loss_local = model.get_loss(
                    atom_type=batch.atom_type,
                    pos=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_nodes_per_graph=batch.num_nodes_per_graph,
                    num_graphs=batch.num_graphs,
                    anneal_power=config.train.anneal_power,
                    return_unreduced_loss=True
                )
                sum_loss += loss.sum().item()
                sum_n += loss.size(0)
                sum_loss_global += loss_global.sum().item()
                sum_n_global += loss_global.size(0)
                sum_loss_local += loss_local.sum().item()
                sum_n_local += loss_local.size(0)
        avg_loss = sum_loss / sum_n
        avg_loss_global = sum_loss_global / sum_n_global
        avg_loss_local = sum_loss_local / sum_n_local
        
        if config.train.scheduler.type == 'plateau':
            scheduler_global.step(avg_loss_global)
            scheduler_local.step(avg_loss_local)
        else:
            scheduler_global.step()
            scheduler_local.step()

        logger.info('[Validate] Iter %05d | Loss %.6f | Loss(Global) %.6f | Loss(Local) %.6f' % (
            it, avg_loss, avg_loss_global, avg_loss_local,
        ))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_global', avg_loss_global, it)
        writer.add_scalar('val/loss_local', avg_loss_local, it)
        writer.flush()
        return avg_loss

    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:     # 经过 val_freq 或全部一轮以后进入验证集
                avg_val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer_global': optimizer_global.state_dict(),
                    'scheduler_global': scheduler_global.state_dict(),
                    'optimizer_local': optimizer_local.state_dict(),
                    'scheduler_local': scheduler_local.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')

