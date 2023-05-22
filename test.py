import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict

from models.epsnet import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *


def num_confs(num:str):
    if num.endswith('x'):
        return lambda x:x*int(num[:-1])
    elif int(num) > 0: 
        return lambda x:int(num)
    else:
        raise ValueError()


if __name__ == '__main__':
    print("here")
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='/home/ycmeng/myc/logs/qm9_default_2023_04_17__07_38_45/checkpoints/3000000.pt', help='path for loading the checkpoint')
    parser.add_argument('--save_traj', action='store_true', default=True,
                    help='whether store the whole trajectory for sampling')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=800)
    parser.add_argument('--end_idx', type=int, default=900)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=5000,
                    help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
                    help='enable global gradients only when noise is low')
    parser.add_argument('--w_global', type=float, default=1.0,
                    help='weight for global gradients')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='ld',
                    help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                    help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()
    print("here")

    # Load checkpoint
    ckpt = torch.load(args.ckpt)
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), '*.yml'))[0]
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))   
    seed_all(config.train.seed)                                         #TODO：seed是啥意思？  单纯的随机数种子
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))               #返回checkpoint所在的上上级目录 在这里为'myc/logs/qm9_default_2022_10_13__17_24_48'
    print("here")

    # Logging
    output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)       #返回结果为 log_dir/sample_时间戳.tag  
    logger = get_logger('test', output_dir)                             #返回一个logger，名为test，文件输出目标为output_dir
    logger.info(args)                                                   #将args通过logger输出到命令行和文件
    print("here")

    # Datasets and loaders
    logger.info('Loading datasets...')      
    transforms = Compose([                                              #里面2个类，CountNodesPerGraph以及AddHigherOrderEdges
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
    ])
    if args.test_set is None:                                           #如果在命令行定义了test_set则按照命令行选择，否则选择checkpoint中的test_set
        test_set = PackedConformationDataset(config.dataset.test, transform=transforms)
    else:
        test_set = PackedConformationDataset(args.test_set, transform=transforms)
    print("here")

    # Model
    logger.info('Loading model...')
    logger.info('%s' % ckpt['config'].model)
    model = get_model(ckpt['config'].model).to(args.device)             #构建模型
    model.load_state_dict(ckpt['model'])                                #将ckpt['model']中的参数以及缓冲数据复制到model中
    print("here")

    test_set_selected = []                                              #从 test_set 中选择从 start_idx 到 end_idx 这些数据
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx): continue
        test_set_selected.append(data)
        print(data, i)

    done_smiles = set()                                                 #在不选择resume的情况下为空
    results = []
    if args.resume is not None:
        with open(args.resume, 'rb') as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)
    print(test_set_selected[0])

    for i, data in enumerate(tqdm(test_set_selected)):
        if data.smiles in done_smiles:                                  #如果在测试中的分则已经有存放入done_similes里面的部分，则认为其已经完成。
            logger.info('Molecule#%d is already done.' % i)
            continue

        num_refs = data.pos_ref.size(0) // data.num_nodes
        num_samples = args.num_confs(num_refs)
        
        data_input = data.clone()
        data_input['pos_ref'] = None
        batch = repeat_data(data_input, num_samples).to(args.device)    #复制一份data然后传到device上
        print('batch.num_graphs')
        print(batch.num_graphs)
        clip_local = None
        for _ in range(2):  # Maximum number of retry
            try:
                pos_init = torch.randn(batch.num_nodes, 3).to(args.device)      
                pos_gen, pos_gen_traj = model.langevin_dynamics_sample(
                    atom_type=batch.atom_type,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False, # Done in transforms.
                    n_steps=args.n_steps,
                    step_lr=1e-6,
                    w_global=args.w_global,
                    global_start_sigma=args.global_start_sigma,
                    clip=args.clip,
                    clip_local=clip_local,
                    sampling_type=args.sampling_type,
                    eta=args.eta
                )
                pos_gen = pos_gen.cpu()
                if args.save_traj:
                    data.pos_gen = torch.stack(pos_gen_traj)
                else:
                    data.pos_gen = pos_gen
                results.append(data)
                done_smiles.add(data.smiles)

                save_path = os.path.join(output_dir, 'samples_%d.pkl' % i)
                logger.info('Saving samples to: %s' % save_path)
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)   #将results保存到f中

                break   # No errors occured, break the retry loop
            except FloatingPointError:
                clip_local = 20
                logger.warning('Retrying with local clipping.')

    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info('Saving samples to: %s' % save_path)

    def get_mol_key(data):
        for i, d in enumerate(test_set_selected):
            if d.smiles == data.smiles:
                return i
        return -1
    results.sort(key=get_mol_key)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
        
    
