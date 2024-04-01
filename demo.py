import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import zlib
import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn_limit, collation_fn_voxelmean, collation_fn_voxelmean_tta,collation_fn_voxelmean_tta_test
from util.logger import get_logger
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, Constant

from util.nuscenes import nuScenes
from util.semantic_kitti import SemanticKITTI
from util.waymo import Waymo

from functools import partial
import pickle
import yaml
from torch_scatter import scatter_mean
import spconv.pytorch as spconv
from model.unet_spherical_transformer import Semantic as Model

if __name__ == "__main__":
    import argparse
    from util import config

    def get_parser():
        parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
        parser.add_argument('--config', type=str, default='config/s3dis/s3dis_stratified_transformer.yaml', help='config file')
        parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
        args = parser.parse_args()
        assert args.config is not None
        cfg = config.load_cfg_from_cfg_file(args.config)
        if args.opts is not None:
            cfg = config.merge_cfg_from_list(cfg, args.opts)
        return cfg

    args = get_parser()

    if args.arch == 'unet_spherical_transformer':
        from model.unet_spherical_transformer import Semantic as Model
        
        args.patch_size = np.array([args.voxel_size[i] * args.patch_size for i in range(3)]).astype(np.float32)
        window_size = args.patch_size * args.window_size
        window_size_sphere = np.array(args.window_size_sphere)
        model = Model(input_c=args.input_c, 
            m=args.m,
            classes=args.classes, 
            block_reps=args.block_reps, 
            block_residual=args.block_residual, 
            layers=args.layers, 
            window_size=window_size, 
            window_size_sphere=window_size_sphere, 
            quant_size=window_size / args.quant_size_scale, 
            quant_size_sphere=window_size_sphere / args.quant_size_scale, 
            rel_query=args.rel_query, 
            rel_key=args.rel_key, 
            rel_value=args.rel_value, 
            drop_path_rate=args.drop_path_rate, 
            window_size_scale=args.window_size_scale, 
            grad_checkpoint_layers=args.grad_checkpoint_layers, 
            sphere_layers=args.sphere_layers,
            a=args.a,
        )
    device = "cuda:1"
    coord =  torch.rand(93726,4)
    xyz = torch.rand(93726,3)
    feat = torch.rand(93726,4)
    offset = torch.tensor([93726])
    target = torch.rand(123389)
    inds_reconstruct = torch.rand(123389)

    inds_reconstruct = inds_reconstruct.cuda(non_blocking=True,device="cuda:1")

    offset_ = offset.clone()
    offset_[1:] = offset_[1:] - offset_[:-1]
    batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

    coord = torch.cat([batch.unsqueeze(-1), coord], -1)
    spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
    coord, xyz, feat, target, offset = coord.cuda(non_blocking=True,device=device), xyz.cuda(non_blocking=True,device=device), feat.cuda(non_blocking=True,device=device), target.cuda(non_blocking=True,device=device), offset.cuda(non_blocking=True,device=device)
    batch = batch.cuda(non_blocking=True,device=device)

    sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size)

    # 设置ONNX文件的保存路径
    output_onnx = './model.onnx'

    # 调用导出函数
    torch.onnx.export(model,               # 模型实例
                    (sinput, xyz, batch),    # 模型输入（注意将输入打包成元组）
                    output_onnx,         # 输出文件名
                    export_params=True,  # 导出模型参数
                    opset_version=12,    # 设置ONNX操作集版本
                    do_constant_folding=True,  # 是否执行常量折叠优化
                    input_names=['input', 'xyz', 'batch'],  # 输入名
                    output_names=['output'],  # 输出名
                    dynamic_axes={'input' : {0 : 'batch_size'},  # 如果有动态批处理大小，指定它
                                    'output' : {0 : 'batch_size'}})