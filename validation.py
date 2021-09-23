import argparse
import yaml
from easydict import EasyDict as edict
import os

import trainer

def attatch_to_config(opt, yaml_args):
    # Pre-train, saving, and loading parameters
    opt.network = yaml_args.name
    opt.load_name = yaml_args.load_name
    opt.vgg_name = yaml_args.vgg_name
    # Training parameters
    opt.saveroot = yaml_args.Validation.saveroot
    opt.whether_save = yaml_args.Validation.whether_save
    opt.val_batch_size = yaml_args.Validation.val_batch_size
    opt.num_workers = yaml_args.Validation.num_workers
    # Initialization parameters
    opt.pad = yaml_args.Network.pad
    opt.activ_g = yaml_args.Network.activ_g
    opt.activ_d = yaml_args.Network.activ_d
    opt.norm = yaml_args.Network.norm
    opt.in_channels = yaml_args.Network.in_channels
    opt.out_channels = yaml_args.Network.out_channels
    opt.start_channels = yaml_args.Network.start_channels
    # Dataset parameters
    opt.baseroot_val = yaml_args.Dataset.baseroot_val
    opt.baseroot_ref = yaml_args.Dataset.baseroot_ref
    opt.small_dataset = yaml_args.Dataset.small_dataset
    opt.ref = yaml_args.Dataset.ref

if __name__ == "__main__":
    
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--yaml_path', type = str, \
        default = './options/mnlc_nr_small.yaml', \
            help = 'yaml_path')
    parser.add_argument('--network', type = str, default = 'MNLC', help = 'network name')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--vgg_name', type = str, default = '', help = 'load the pre-trained vgg model with certain epoch')
    # Validation parameters
    parser.add_argument('--saveroot', type = str, default = '', help = 'saveroot')
    parser.add_argument('--whether_save', type = bool, default = True, help = 'whether_save')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'num_workers')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 1, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--out_channels', type = int, default = 3, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    # Dataset parameters
    parser.add_argument('--baseroot_val', type = str, \
        default = 'E:\\submitted papers\\QuadBayer Deblur\\data\\val', \
            help = 'output image baseroot')
    parser.add_argument('--baseroot_ref', type = str, \
        default = 'E:\\submitted papers\\QuadBayer Deblur\\data\\ref', \
            help = 'output image baseroot')
    parser.add_argument('--small_dataset', type = bool, default = True, help = 'small_dataset')
    parser.add_argument('--ref', type = bool, default = False, help = 'ref')
    opt = parser.parse_args()

    with open(opt.yaml_path, mode = 'r') as f:
        yaml_args = edict(yaml.load(f))
    
    attatch_to_config(opt, yaml_args)
    print(opt)

    trainer.Valer(opt)
    