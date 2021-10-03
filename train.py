import argparse
import yaml
from easydict import EasyDict as edict
import os

import trainer

def attatch_to_config(opt, yaml_args):
    # Pre-train, saving, and loading parameters
    opt.network = yaml_args.name
    opt.save_path = yaml_args.Training.save_path
    opt.sample_path = yaml_args.Training.sample_path
    opt.save_mode = yaml_args.Training.save_mode
    opt.save_by_epoch = yaml_args.Training.save_by_epoch
    opt.save_by_iter = yaml_args.Training.save_by_iter
    opt.load_name = ""
    opt.vgg_name = yaml_args.vgg_name
    opt.multi_gpu = yaml_args.Training.multi_gpu
    opt.cudnn_benchmark = yaml_args.Training.cudnn_benchmark
    # Training parameters
    opt.epochs = yaml_args.Training.epochs
    opt.train_batch_size = yaml_args.Training.train_batch_size
    opt.val_batch_size = yaml_args.Training.val_batch_size
    opt.lr_g = yaml_args.Training.lr_g
    opt.lr_d = yaml_args.Training.lr_d
    opt.b1 = yaml_args.Training.b1
    opt.b2 = yaml_args.Training.b2
    opt.weight_decay = yaml_args.Training.weight_decay
    opt.lr_decrease_epoch = yaml_args.Training.lr_decrease_epoch
    opt.num_workers = yaml_args.Training.num_workers
    opt.lambda_l1 = yaml_args.Training.lambda_l1
    opt.lambda_p = yaml_args.Training.lambda_p
    opt.lambda_s = yaml_args.Training.lambda_s
    opt.lambda_gan = yaml_args.Training.lambda_gan
    opt.lambda_tr = yaml_args.Training.lambda_tr
    opt.margin_tr = yaml_args.Training.margin_tr
    # Initialization parameters
    opt.init_type = yaml_args.Network.init_type
    opt.init_gain = yaml_args.Network.init_gain
    # Dataset parameters
    opt.baseroot_train = yaml_args.Dataset.baseroot_train
    opt.baseroot_val = yaml_args.Dataset.baseroot_val
    opt.noise_type = yaml_args.Dataset.noise_type
    opt.trans_type = yaml_args.Dataset.trans_type
    opt.a = yaml_args.Dataset.a
    opt.b = yaml_args.Dataset.b
    opt.mean = yaml_args.Dataset.mean
    opt.std = yaml_args.Dataset.std

if __name__ == "__main__":
    
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--yaml_path', type = str, \
        default = './options/cvpr2020.yaml', \
            help = 'yaml_path')
    parser.add_argument('--network', type = str, default = 'NRC', help = 'network name')
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 100000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--vgg_name', type = str, default = '', help = 'load the pre-trained vgg model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 300, help = 'number of epochs of training')
    parser.add_argument('--train_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0001, help = 'Adam: learning rate for G / D')
    parser.add_argument('--lr_d', type = float, default = 0.0001, help = 'Adam: learning rate for G / D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 150, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_l1', type = float, default = 30, help = 'coefficient for GAN Loss')
    parser.add_argument('--lambda_p', type = float, default = 0.01, help = 'coefficient for GAN Loss')
    parser.add_argument('--lambda_s', type = float, default = 50, help = 'coefficient for GAN Loss')
    parser.add_argument('--lambda_gan', type = float, default = 1, help = 'coefficient for GAN Loss')
    parser.add_argument('--lambda_tr', type = float, default = 0, help = 'coefficient for GAN Loss')
    parser.add_argument('--margin_tr', type = float, default = 12, help = 'margin for triplet Loss')
    # Initialization parameters
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--baseroot_train', type = str, \
        default = 'E:\\submitted papers\\QuadBayer Deblur\\data\\train', \
            help = 'input image baseroot')
    parser.add_argument('--baseroot_val', type = str, \
        default = 'E:\\submitted papers\\QuadBayer Deblur\\data\\val', \
            help = 'output image baseroot')
    parser.add_argument('--noise_type', type = str, default = 'uniform', help = 'uniform | gaussian')
    parser.add_argument('--a', type = float, default = -1, help = 'parameter for uniform noise')
    parser.add_argument('--b', type = float, default = 1, help = 'parameter for uniform noise')
    parser.add_argument('--mean', type = float, default = 0, help = 'parameter for gaussian noise')
    parser.add_argument('--std', type = float, default = 0.01, help = 'parameter for gaussian noise')
    parser.add_argument('--trans_type', type = str, default = 'tps', help = 'tps | elastic')
    opt = parser.parse_args()

    with open(opt.yaml_path, mode = 'r') as f:
        yaml_args = edict(yaml.load(f))
    
    attatch_to_config(opt, yaml_args)
    print(opt)

    trainer.Simple_Trainer(opt)
    #trainer.Trainer(opt)
    