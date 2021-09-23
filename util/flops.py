import argparse
import torch
import yaml
from easydict import EasyDict as edict
from thop import profile
from thop import clever_format

import network

def attatch_to_config(opt, yaml_args):
    # Pre-train, saving, and loading parameters
    opt.network = yaml_args.name
    # Initialization parameters
    opt.pad = yaml_args.Network.pad
    opt.activ_g = yaml_args.Network.activ_g
    opt.activ_d = yaml_args.Network.activ_d
    opt.norm = yaml_args.Network.norm
    opt.in_channels = yaml_args.Network.in_channels
    opt.out_channels = yaml_args.Network.out_channels
    opt.start_channels = yaml_args.Network.start_channels
    opt.init_type = yaml_args.Network.init_type
    opt.init_gain = yaml_args.Network.init_gain

def create_generator_for_flops(opt):
    # Initialize the network
    generator = getattr(network, opt.network)(opt)
    network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
    return generator

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters (should be changed)
    parser.add_argument('--yaml_path', type = str, \
        #default = './options/denoising/ircnn_rgb1.yaml', \
        #default = './options/deblurring/deepdeblur_rgb1.yaml', \
        #default = './options/joint/pix2pix_raw1.yaml', \
        default = './options/joint/lsd2_raw1.yaml', \
            help = 'yaml_path')
    # Initialization parameters (just ignore all of them since parameters are recorded in option.yaml)
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--out_channels', type = int, default = 3, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Flops parameters (should be changed)
    parser.add_argument('--h', type = int, default = 512, help = 'h resolution')
    parser.add_argument('--w', type = int, default = 512, help = 'w resolution')
    opt = parser.parse_args()

    # sgn_deblurnet_v1_001: macs: 18.963G params: 4.904M
    # sgn_deblurnet_v1_002: macs: 21.649G params: 5.273M
    # sgn_deblurnet_v1_011: macs: 5.266G params: 1.480M
    # sgn_deblurnet_v1_012: macs: 5.938G params: 1.573M
    # sgn_deblurnet_v1_021: macs: 3.156G params: 927.843K
    # sgn_deblurnet_v1_022: macs: 3.534G params: 979.827K

    # tp_denoisenet_v1_001: macs: 79.629G params: 5.410M

    with open(opt.yaml_path, mode = 'r') as f:
        yaml_args = edict(yaml.load(f))

    attatch_to_config(opt, yaml_args)

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    
    # Define the network
    generator = create_generator_for_flops(opt).cuda()

    for param in generator.parameters():
        param.requires_grad = False

    # forward propagation
    if 'raw' in opt.yaml_path:
        input = torch.randn(1, 1, opt.h, opt.w).cuda()
    if 'rgb' in opt.yaml_path:
        input = torch.randn(1, 3, opt.h, opt.w).cuda()

    macs, params = profile(generator, inputs = (input, ))
    macs_1, params_1 = clever_format([macs, params], "%.3f")
    print('name:', opt.yaml_path, 'macs:', macs_1, 'params:', params_1)
