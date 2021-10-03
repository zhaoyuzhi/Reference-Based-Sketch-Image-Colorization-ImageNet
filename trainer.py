import time
import datetime
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils
from util import gram

def Simple_Trainer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    tail_name = '%s_%s_%s' % (opt.network, opt.noise_type, opt.trans_type)
    save_folder = os.path.join(opt.save_path, tail_name)
    sample_folder = os.path.join(opt.sample_path, tail_name)
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet_multilayer(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = '%s_epoch%d_bs%d.pth' % (opt.network, epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = '%s_iter%d_bs%d.pth' % (opt.network, iteration, opt.train_batch_size)
        save_model_path = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    trainset = dataset.ImageNet_Dataset(opt)
    print('The overall number of training images:', len(trainset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, (grayscale, ref, rgb) in enumerate(train_loader):

            # To device
            grayscale = grayscale.cuda()
            ref = ref.cuda()
            rgb = rgb.cuda()

            ### Train Generator
            optimizer_G.zero_grad()
            out, q_k_v_list = generator(ref, grayscale)
            
            # L1 Loss
            loss_L1 = criterion_L1(out, rgb)

            # Perceptual Loss
            conv2_2_out, conv3_3_out, conv4_3_out, conv5_3_out = perceptualnet(out)
            conv2_2_rgb, conv3_3_rgb, conv4_3_rgb, conv5_3_rgb = perceptualnet(rgb)
            loss_p = criterion_L1(conv4_3_out, conv4_3_rgb) + criterion_L1(conv5_3_out, conv5_3_rgb)
            
            # Overall Loss and optimize
            loss = opt.lambda_l1 * loss_L1 + opt.lambda_p * loss_p
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Pecep Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(train_loader), loss_L1.item(), loss_p.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)
            
        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            if grayscale.shape[1] == 1:
                grayscale = torch.cat((grayscale, grayscale, grayscale), 1)
            img_list = [grayscale, ref, out, rgb]
            name_list = ['grayscale', 'ref', 'out', 'rgb']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_iter%d' % (epoch + 1), \
                img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
        
        '''
        ### Validation
        val_PSNR = 0
        num_of_val_image = 0

        for j, (true_input, true_target) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Forward propagation
            with torch.no_grad():
                fake_target = generator(true_input)

            # Accumulate num of image and val_PSNR
            num_of_val_image += true_input.shape[0]
            val_PSNR += utils.psnr(fake_target, true_target, 1) * true_input.shape[0]
        val_PSNR = val_PSNR / num_of_val_image

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [fake_target, true_target]
            name_list = ['pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'val_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Record average PSNR
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))
        '''

def Trainer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    tail_name = '%s_%s_%s' % (opt.network, opt.noise_type, opt.trans_type)
    save_folder = os.path.join(opt.save_path, tail_name)
    sample_folder = os.path.join(opt.sample_path, tail_name)
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_triplet = nn.TripletMarginLoss(margin = opt.margin_tr).cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet_multilayer(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = '%s_epoch%d_bs%d.pth' % (opt.network, epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = '%s_iter%d_bs%d.pth' % (opt.network, iteration, opt.train_batch_size)
        save_model_path = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    trainset = dataset.ImageNet_Dataset(opt)
    print('The overall number of training images:', len(trainset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, (grayscale, ref, rgb) in enumerate(train_loader):

            # To device
            grayscale = grayscale.cuda()
            ref = ref.cuda()
            rgb = rgb.cuda()

            ### Train Discriminator
            optimizer_D.zero_grad()
            out, q_k_v_list = generator(ref, grayscale)
            
            # Fake colorizations
            fake_scalar_d = discriminator(out.detach(), grayscale)

            # True colorizations
            true_scalar_d = discriminator(rgb, grayscale)

            # Overall Loss and optimize
            loss_D = - torch.mean(true_scalar_d) + torch.mean(fake_scalar_d)
            loss_D.backward()
            optimizer_D.step()

            ### Train Generator
            optimizer_G.zero_grad()
            out, q_k_v_list = generator(ref, grayscale)
            
            # L1 Loss
            loss_L1 = criterion_L1(out, rgb)

            # Perceptual Loss
            conv2_2_out, conv3_3_out, conv4_3_out, conv5_3_out = perceptualnet(out)
            conv2_2_rgb, conv3_3_rgb, conv4_3_rgb, conv5_3_rgb = perceptualnet(rgb)
            loss_p = criterion_L1(conv4_3_out, conv4_3_rgb)
            
            # Style Loss
            gram4_3_out = gram.gram_matrix(conv4_3_out)
            gram4_3_rgb = gram.gram_matrix(conv4_3_rgb)
            loss_s = criterion_L1(gram4_3_out, gram4_3_rgb)

            # GAN Loss
            fake_scalar = discriminator(out, grayscale)
            loss_GAN = - torch.mean(fake_scalar)

            # Triplet Loss
            if opt.lambda_tr > 0:
                batch_size = grayscale.shape[0]
                anchor = q_k_v_list[0].view(batch_size, -1)
                positive = q_k_v_list[1].contiguous().view(batch_size, -1)
                negative = q_k_v_list[2].contiguous().view(batch_size, -1)
                loss_triple = criterion_triplet(anchor = anchor, positive = positive, negative = negative)

            # Overall Loss and optimize
            loss = opt.lambda_l1 * loss_L1 + opt.lambda_p * loss_p + opt.lambda_s * loss_s + opt.lambda_gan * loss_GAN
            if opt.lambda_tr > 0:
                loss = loss + opt.lambda_tr * loss_triple
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if opt.lambda_tr > 0:
                print("\r[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Pecep Loss: %.4f] [Style Loss: %.4f] [GAN Loss: %.4f] [Tri Loss: %.4f] Time_left: %s" %
                    ((epoch + 1), opt.epochs, i, len(train_loader), loss_L1.item(), loss_p.item(), loss_s.item(), loss_GAN.item(), loss_triple.item(), time_left))
            else:
                print("\r[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Pecep Loss: %.4f] [Style Loss: %.4f] [GAN Loss: %.4f] Time_left: %s" %
                    ((epoch + 1), opt.epochs, i, len(train_loader), loss_L1.item(), loss_p.item(), loss_s.item(), loss_GAN.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)
            
        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            if grayscale.shape[1] == 1:
                grayscale = torch.cat((grayscale, grayscale, grayscale), 1)
            img_list = [grayscale, ref, out, rgb]
            name_list = ['grayscale', 'ref', 'out', 'rgb']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_iter%d' % (epoch + 1), \
                img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
        
        '''
        ### Validation
        val_PSNR = 0
        num_of_val_image = 0

        for j, (true_input, true_target) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Forward propagation
            with torch.no_grad():
                fake_target = generator(true_input)

            # Accumulate num of image and val_PSNR
            num_of_val_image += true_input.shape[0]
            val_PSNR += utils.psnr(fake_target, true_target, 1) * true_input.shape[0]
        val_PSNR = val_PSNR / num_of_val_image

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [fake_target, true_target]
            name_list = ['pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'val_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Record average PSNR
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))
        '''

def Valer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # configurations
    tail_name = '%s_%s_%s' % (opt.network, opt.noise_type, opt.trans_type)
    save_folder = os.path.join(opt.saveroot, tail_name)
    if opt.whether_save:
        utils.check_path(save_folder)

    # Initialize Generator
    opt.load_name = os.path.join(opt.load_name, tail_name, opt.network + '_epoch_40_bs64.pth')
    generator = utils.create_generator(opt)

    # To device
    generator = generator.cuda()
    
    # Define the dataset
    valset = dataset.ImageNet_ValDataset(opt)
    print('The overall number of validation images:', len(valset))

    # Define the dataloader
    val_loader = DataLoader(valset, batch_size = opt.val_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # forward
    val_PSNR = 0
    val_SSIM = 0
    
    # For loop training
    for i, (grayscale, ref, rgb, save_img_path) in enumerate(val_loader):

        # To device
        grayscale = grayscale.cuda()
        ref = ref.cuda()
        rgb = rgb.cuda()
        save_img_path = save_img_path[0].split('.')[0] + '.png'
        
        # Train Generator
        out, _ = generator(ref, grayscale)
        
        # Save the image (BCHW -> HWC)
        if opt.whether_save:
            save_img = out[0, :, :, :].clone().data.permute(1, 2, 0).cpu().numpy()
            save_img = np.clip(save_img, 0, 1)
            save_img = (save_img * 255).astype(np.uint8)
            save_full_path = os.path.join(save_folder, save_img_path)
            cv2.imwrite(save_full_path, save_img)

        # PSNR
        # print('The %d-th image PSNR %.4f' % (i, val_PSNR_this))
        this_PSNR = utils.psnr(out, rgb, 1) * rgb.shape[0]
        val_PSNR += this_PSNR
        this_SSIM = utils.ssim(out, rgb) * rgb.shape[0]
        val_SSIM += this_SSIM
        print('The %d-th image: Name: %s PSNR: %.5f, SSIM: %.5f' % (i + 1, save_img_path, this_PSNR, this_SSIM))

    val_PSNR = val_PSNR / len(valset)
    val_SSIM = val_SSIM / len(valset)
    print('The average of %s: PSNR: %.5f, average SSIM: %.5f' % (opt.load_name, val_PSNR, val_SSIM))
