# Reference-Based-Sketch-Image-Colorization-ImageNet

This is a PyTorch implementation of CVPR 2020 paper "Reference-Based Sketch Image Colorization using Augmented-Self Reference and Dense Semantic Correspondence" (https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Reference-Based_Sketch_Image_Colorization_Using_Augmented-Self_Reference_and_Dense_Semantic_CVPR_2020_paper.pdf)

We will provide pre-trained model on ImageNet dataset shortly

## 1 Training

- Prepare the ImageNet dataset (i.e., upload ILSVRC2012_train_256 folder to your server)

- Download the PyTorch official pre-trained VGG-16 model, and then rename it to 'vgg16_pretrained.pth'

(torchvision webpage: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)

(download webpage: https://download.pytorch.org/models/vgg16-397923af.pth) (this is good)

- Change the parameter in yaml file and run

(--vgg_name -> your VGG-16 model path)

(--baseroot_train -> your ImageNet dataset path, i.e., ILSVRC2012_train_256 path)

```bash
sh sbatch_run.sh or sh local_run.sh
```

By the way, I use 8 Titan GPUs to train the network with batch size of 32, epoch of 40. It takes approximately 16 days!

The forward of GAN discriminator and VGG-16 take a lot of time, which are used to compute GAN loss and perceptual loss, etc.

## 2 Validation

- Prepare the references with same names to ImageNet test10k

- Change the parameter in yaml file and run

```bash
sh val_run.sh or sh validation.sh
```
