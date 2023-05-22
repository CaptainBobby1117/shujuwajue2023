import os
import argparse
import pickle
import yaml
import torch
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict

from models.epsnet import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *


# d = "/home/ycmeng/myc/data/GEOM/QM9/test_data_1k.pkl"
# with open(d, "rb") as f:
#     data = pickle.load(f)
# print(data[900])

# Load checkpoint
# ckpt = torch.load('/home/ycmeng/myc/logs/qm9_default_2023_04_17__07_38_45/checkpoints/3000000.pt')
# config_path = glob(os.path.join(os.path.dirname(os.path.dirname('/home/ycmeng/myc/logs/qm9_default_2023_04_17__07_38_45/checkpoints/3000000.pt')), '*.yml'))[0]
# with open(config_path, 'r') as f:
#     config = EasyDict(yaml.safe_load(f))   
# seed_all(config.train.seed)                                         #TODO：seed是啥意思？  单纯的随机数种子
# log_dir = os.path.dirname(os.path.dirname('/home/ycmeng/myc/logs/qm9_default_2023_04_17__07_38_45/checkpoints/3000000.pt'))               #返回checkpoint所在的上上级目录 在这里为'myc/logs/qm9_default_2022_10_13__17_24_48'
# print("here")
# print('config.model.edge_order')
# print(config.model.edge_order)

# transforms = Compose([                                              #里面2个类，CountNodesPerGraph以及AddHigherOrderEdges
#     CountNodesPerGraph(),
#     AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
# ])    
# test_set = PackedConformationDataset('data/GEOM/QM9/test_data_1k.pkl', transform=transforms)
# print(test_set[0])

# d = "/home/ycmeng/myc/logs/qm9_default_2023_04_17__07_38_45/sample_2023_04_18__14_39_10/samples_100.pkl"
# with open(d, "rb") as f:
#     data = pickle.load(f)
# print(data[100])
# print(data[100].pos_gen.size())
# print(data[100].pos_gen)
# print(data[100].pos_ref.size())
# print(data[100].pos_ref)
# print(data[100].pos.size())
# print(data[100].pos)

# print(data[0])
# print(data[0].pos_gen.size())
# print(data[0].pos_gen)
# print(data[0].pos_ref.size())
# print(data[0].pos_ref)
# print(data[0].pos.size())
# print(data[0].pos)
# k = []
# d1 = data[0].pos.clone()
# k.append(d1)
# d2 = data[0].pos.clone()
# k.append(d2)
# m=torch.stack(k)
# print(m.size())


# path = 'data/GEOM/QM9/test_data_1k.pkl'
# with open(path, 'rb') as f:
#     data = pickle.load(f)
#     print(hasattr(data, 'idx'))
    # sm = {}
    # id = {}
    # for d in data:
    #     if d.smiles not in sm.keys():
    #         sm[d.smiles] = 1
    #     else:
    #         sm[d.smiles] += 1
    #     if d.idx not in id.keys():
    #         id[d.idx] = 1
    #     else:
    #         id[d.idx] += 1
    # for key in sm.keys():
    #     if sm[key] != 1:
    #         print(key, sm[key])
    # for key in id.keys():
    #     if id[key] != 1:
    #         print(key, id[key])
    # print(len(sm.keys()))
    # print(len(id.keys()))

d = "/home/ycmeng/myc/logs/qm9_default_2023_04_17__07_38_45/sample_2023_05_13__09_50_18/samples_0.pkl"
with open(d, "rb") as f:
    data = pickle.load(f)
print(data)
for i, pos in enumerate(data[0].pos_gen):
    #os.remove('gen_%d.npy' % i)
    x = pos[:len(data[0].atom_type)].numpy()
    np.save('./sample_new/gen_%d' % i, x)
np.save('./sample_new/gen_5000', data[0].pos_ref.numpy())
