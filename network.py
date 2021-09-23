import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision as tv
import math

from util.block import *
import utils

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                Generator
# ----------------------------------------
class ResBlockNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlockNet, self).__init__()
        self.main = list()
        self.main.append(ResBlock(in_channels, out_channels))
        self.main.append(ResBlock(out_channels, out_channels))
        self.main.append(ResBlock(out_channels, out_channels))
        self.main.append(ResBlock(out_channels, out_channels))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x) + x


class Encoder(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channels=3, spec_norm=False, LR=0.2):
        super(Encoder, self).__init__()

        self.layer1 = ConvBlock(in_channels, 16, spec_norm, LR=LR) # 256
        self.layer2 = ConvBlock(16, 16, spec_norm, LR=LR) # 256
        self.layer3 = ConvBlock(16, 32, spec_norm, stride=2, LR=LR) # 128
        self.layer4 = ConvBlock(32, 32, spec_norm, LR=LR) # 128
        self.layer5 = ConvBlock(32, 64, spec_norm, stride=2, LR=LR) # 64
        self.layer6 = ConvBlock(64, 64, spec_norm, LR=LR) # 64
        self.layer7 = ConvBlock(64, 128, spec_norm, stride=2, LR=LR) # 32
        self.layer8 = ConvBlock(128, 128, spec_norm, LR=LR) # 32
        self.layer9 = ConvBlock(128, 256, spec_norm, stride=2, LR=LR) # 16
        self.layer10 = ConvBlock(256, 256, spec_norm, LR=LR) # 16
        self.down_sampling = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):

        feature_map1 = self.layer1(x)
        feature_map2 = self.layer2(feature_map1)
        feature_map3 = self.layer3(feature_map2)
        feature_map4 = self.layer4(feature_map3)
        feature_map5 = self.layer5(feature_map4)
        feature_map6 = self.layer6(feature_map5)
        feature_map7 = self.layer7(feature_map6)
        feature_map8 = self.layer8(feature_map7)
        feature_map9 = self.layer9(feature_map8)
        feature_map10 = self.layer10(feature_map9)

        down_feature_map1 = self.down_sampling(feature_map1)
        down_feature_map2 = self.down_sampling(feature_map2)
        down_feature_map3 = self.down_sampling(feature_map3)
        down_feature_map4 = self.down_sampling(feature_map4)
        down_feature_map5 = self.down_sampling(feature_map5)
        down_feature_map6 = self.down_sampling(feature_map6)
        down_feature_map7 = self.down_sampling(feature_map7)
        down_feature_map8 = self.down_sampling(feature_map8)

        #print("feature_map1.size : ", feature_map1.size()) # torch.Size([2, 16, 256, 256])
        #print("feature_map2.size : ", feature_map2.size()) # torch.Size([2, 16, 256, 256])
        #print("feature_map3.size : ", feature_map3.size()) # torch.Size([2, 32, 128, 128])
        #print("feature_map4.size : ", feature_map4.size()) # torch.Size([2, 32, 128, 128])
        #print("feature_map5.size : ", feature_map5.size()) # torch.Size([2, 64, 64, 64])
        #print("feature_map6.size : ", feature_map6.size()) # torch.Size([2, 64, 64, 64])
        #print("feature_map7.size : ", feature_map7.size()) # feature_map7.size :  torch.Size([2, 128, 32, 32])
        #print("feature_map8.size : ", feature_map8.size()) # feature_map7.size :  torch.Size([2, 128, 32, 32])
        #print("feature_map9.size : ", feature_map9.size()) # feature_map9.size :  torch.Size([2, 256, 16, 16])
        #print("feature_map10.size : ", feature_map10.size()) # feature_map9.size :  torch.Size([2, 256, 16, 16])

        output = torch.cat([down_feature_map1,
                            down_feature_map2,
                            down_feature_map3,
                            down_feature_map4,
                            down_feature_map5,
                            down_feature_map6,
                            down_feature_map7,
                            down_feature_map8,
                            feature_map9,
                            feature_map10,
                            ], dim=1)

        feature_list = [feature_map1,
         feature_map2,
         feature_map3,
         feature_map4,
         feature_map5,
         feature_map6,
         feature_map7,
         feature_map8,
         feature_map9,
         #feature_map10,
         ]
        #print('output.size : ', output.size()) # output.size :  torch.Size([2, 992, 16, 16])
        b, ch, h, w = output.size()
        #output = output.reshape((b, ch, h * w)) # output.size :  torch.Size([2, 992, 256])
        output = output.reshape((b, h * w, ch)) # output.size :  torch.Size([2, 256, 992])
        #print('output.size : ', output.size())
        return output, feature_list

class Decoder(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=False, LR=0.2):
        super(Decoder, self).__init__()
        self.layer10 = ConvBlock(992 * 2, 256, spec_norm, LR=LR) # 16->16
        self.layer9 = ConvBlock(256 + 256, 256, spec_norm, LR=LR) # 16->16
        self.layer8 = ConvBlock(256 + 128, 128, spec_norm, LR=LR, up=True) # 16->32
        self.layer7 = ConvBlock(128 + 128, 128, spec_norm, LR=LR) # 32->32
        self.layer6 = ConvBlock(128 + 64, 64, spec_norm, LR=LR, up=True) # 32-> 64
        self.layer5 = ConvBlock(64 + 64, 64, spec_norm, LR=LR) # 64 -> 64
        self.layer4 = ConvBlock(64 + 32, 32, spec_norm, LR=LR, up=True) # 64 -> 128
        self.layer3 = ConvBlock(32 + 32, 32, spec_norm, LR=LR) # 128 -> 128
        self.layer2 = ConvBlock(32 + 16, 16, spec_norm, LR=LR, up=True) # 128 -> 256
        self.layer1 = ConvBlock(16 + 16, 16, spec_norm, LR=LR) # 256 -> 256
        self.last_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, feature_list):

        feature_map10 = self.layer10(x)
        feature_map9 = self.layer9(torch.cat([feature_map10, feature_list[-1]], dim=1))
        feature_map8 = self.layer8(feature_map9, feature_list[-2])
        feature_map7 = self.layer7(torch.cat([feature_map8, feature_list[-3]], dim=1))
        feature_map6 = self.layer6(feature_map7, feature_list[-4])
        feature_map5 = self.layer5(torch.cat([feature_map6, feature_list[-5]], dim=1))
        feature_map4 = self.layer4(feature_map5, feature_list[-6])
        feature_map3 = self.layer3(torch.cat([feature_map4, feature_list[-7]], dim=1))
        feature_map2 = self.layer2(feature_map3, feature_list[-8])
        feature_map1 = self.layer1(torch.cat([feature_map2, feature_list[-9]], dim=1))
        feature_map0 = self.last_conv(feature_map1)
        """
        print('x : {}, feature_map10 : {}'.format(x.size(), feature_map10.size()))
        print('feature_map9 : {}, feature_list[-1] : {}'.format(feature_map9.size(), feature_list[-1].size()))
        print('feature_map8 : {}, feature_list[-2] : {}'.format(feature_map8.size(), feature_list[-2].size()))
        print('feature_map7 : {}, feature_list[-3] : {}'.format(feature_map7.size(), feature_list[-3].size()))
        print('feature_map6 : {}, feature_list[-4] : {}'.format(feature_map6.size(), feature_list[-4].size()))
        print('feature_map5 : {}, feature_list[-5] : {}'.format(feature_map5.size(), feature_list[-5].size()))
        print('feature_map4 : {}, feature_list[-6] : {}'.format(feature_map4.size(), feature_list[-6].size()))
        print('feature_map3 : {}, feature_list[-7] : {}'.format(feature_map3.size(), feature_list[-7].size()))
        print('feature_map2 : {}, feature_list[-8] : {}'.format(feature_map2.size(), feature_list[-8].size()))
        print('feature_map1 : {}, feature_list[-9] : {}'.format(feature_map1.size(), feature_list[-9].size()))
        feature_map9 : torch.Size([2, 256, 16, 16]), feature_list[-1] : torch.Size([2, 256, 16, 16])
        feature_map8 : torch.Size([2, 128, 32, 32]), feature_list[-2] : torch.Size([2, 128, 32, 32])
        feature_map7 : torch.Size([2, 128, 32, 32]), feature_list[-3] : torch.Size([2, 128, 32, 32])
        feature_map6 : torch.Size([2, 64, 64, 64]), feature_list[-4] : torch.Size([2, 64, 64, 64])
        feature_map5 : torch.Size([2, 64, 64, 64]), feature_list[-5] : torch.Size([2, 64, 64, 64])
        feature_map4 : torch.Size([2, 32, 128, 128]), feature_list[-6] : torch.Size([2, 32, 128, 128])
        feature_map3 : torch.Size([2, 32, 128, 128]), feature_list[-7] : torch.Size([2, 32, 128, 128])
        feature_map2 : torch.Size([2, 16, 256, 256]), feature_list[-8] : torch.Size([2, 16, 256, 256])
        feature_map1 : torch.Size([2, 16, 256, 256]), feature_list[-9] : torch.Size([2, 16, 256, 256])
        """
        return self.tanh(feature_map0)

class SCFT_Module(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self):
        super(SCFT_Module, self).__init__()
        self.w_q = nn.Linear(992, 992)
        self.w_k = nn.Linear(992, 992)
        self.w_v = nn.Linear(992, 992)
        self.scailing_factor = math.sqrt(992)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, v_s, v_r):

        #print('v_s.size() : ', v_s.size()) # v_s.size() :  torch.Size([2, 256, 992])
        #print('v_r.size() : ', v_r.size()) # v_r.size() :  torch.Size([2, 256, 992])
        q_result = self.w_q(v_s)
        k_result = self.w_k(v_r)
        v_result = self.w_v(v_r)

        #print('q_result : ', q_result.size()) # q_result :  torch.Size([2, 256, 992])
        #print('k_result : ', k_result.size()) # k_result :  torch.Size([2, 256, 992])
        k_result = k_result.permute(0, 2, 1)
        #print('k_result : ', k_result.size()) # k_result :  torch.Size([2, 992, 256])
        attention_map = torch.bmm(q_result, k_result)
        #print('attention_map : ', attention_map.size()) # attention_map :  torch.Size([2, 256, 256])
        attention_map = self.softmax(attention_map) / self.scailing_factor
        v_star = torch.bmm(attention_map, v_result)
        #print('v_star.size() : ', v_star.size()) # v_star.size() :  torch.Size([2, 256, 992])
        """
        *debug example
        soft_max = nn.Softmax(dim=2)
        t = torch.randn((2,5,5))
        softed_t = soft_max(t)
        print(softed_t)
        print(torch.sum(softed_t[0, 0]))
        """
        v_sum = (v_s + v_star).permute(0, 2, 1)
        #print('v_sum.size() : ', v_sum.size()) # v_sum.size() :  torch.Size([2, 992, 256])
        b, ch, hw = v_sum.size()
        v_sum = v_sum.reshape((b, ch, 16, 16))
        #print('v_sum.size() : ', v_sum.size())  # v_sum.size() :  torch.Size([2, 992, 16, 16])
        return v_sum, [q_result, k_result, v_result]

class Generator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=False, LR=0.2):
        super(Generator, self).__init__()
        self.encoder_reference = Encoder(in_channels=3, spec_norm=spec_norm, LR=LR)
        self.encoder_sketch = Encoder(in_channels=1, spec_norm=spec_norm, LR=LR)
        self.decoder = Decoder()
        self.scft_module = SCFT_Module()
        self.res_model = ResBlockNet(992, 992)

    def forward(self, reference, sketch): # here sketch is grayscale for ImageNet dataset
        v_r, _ = self.encoder_reference(reference)
        v_s, feature_list = self.encoder_sketch(sketch)
        v_c, q_k_v_list = self.scft_module(v_s, v_r)
        rv_c = self.res_model(v_c)
        concat = torch.cat([rv_c, v_c], dim=1)
        image = self.decoder(concat, feature_list)
        return image, q_k_v_list

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=True, LR=0.2):
        super(Discriminator, self).__init__()
        self.main = list()
        self.main.append(ConvBlock(4, 16, spec_norm, stride=2, LR=LR)) # 256 -> 128
        self.main.append(ConvBlock(16, 32, spec_norm, stride=2, LR=LR)) # 128 -> 64
        self.main.append(ConvBlock(32, 64, spec_norm, stride=2, LR=LR)) # 64 -> 32
        self.main.append(ConvBlock(64, 128, spec_norm, stride=2, LR=LR)) # 32 -> 16
        self.main.append(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*self.main)

    def forward(self, x, grayscale):
        x = torch.cat((x, grayscale), 1)
        return self.main(x)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 1, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--out_channels', type = int, default = 3, help = '1 for quadbayer, 3 for rgb')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--vgg_name', type = str, default = './util/vgg16_pretrained.pth', help = 'load the pre-trained vgg model with certain epoch')
    opt = parser.parse_args()

    # G
    net = Generator(opt).cuda()
    grayscale = torch.randn(1, 1, 256, 256).cuda()
    ref = torch.randn(1, 3, 256, 256).cuda()
    y, q_k_v_list = net(ref, grayscale)
    print(y.shape) # torch.Size([1, 3, 256, 256])
    
    # D
    net = Discriminator(opt).cuda()
    z = net(y, grayscale)
    print(z.shape) # torch.Size([1, 1, 16, 16])

    #torch.save(net.state_dict(), 'test.pth')
    