import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
import utils
import copy


class TabularClassifier(nn.Module):
    def __init__(self, input_dim_classifer, hidden_dim, output_dim_classifer):
        super(TabularClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim_classifer, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim_classifer),
            nn.LogSoftmax(dim=1)  # 使用 LogSoftmax 适用于分类任务0/1
        )

    def forward(self, x):
        return self.model(x)


class TabularGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, good_bad_dim, sensitive_dim, feature_dim):
        super(TabularGenerator, self).__init__()
        self.latent_dims = input_dim
        self.encoder = Encoder(input_dim, self.latent_dims)
        self.decoder = Decoder(self.latent_dims, output_dim)
        self.good_bad_dim = good_bad_dim
        self.sensitive_dim = sensitive_dim
        self.feature_dim = feature_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 批归一化层
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 批归一化层
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # 使用编码器编码输入
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)

        # 使用解码器生成数据
        generated_data = self.decoder(latent)

        # 提取 good_bad, sensitive 和 features
        good_bad = generated_data[:, :self.good_bad_dim]
        good_bad = torch.sigmoid(good_bad)
        good_bad = (good_bad > 0.5).float()
        sensitive = generated_data[:,
                                   self.good_bad_dim:self.good_bad_dim + self.sensitive_dim]
        sensitive = torch.sigmoid(sensitive)
        sensitive = (sensitive > 0.5).float()
        features = generated_data[:, self.good_bad_dim +
                                  self.sensitive_dim:self.good_bad_dim + self.sensitive_dim + self.feature_dim]

        return good_bad, sensitive, features

    def latent_sample(self, mu, logvar):
        # reparameterization trick
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        return eps.mul(std).add_(mu)


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dims=5):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dims = latent_dims
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)  # 批归一化层
        self.fc_mu = nn.Linear(64, latent_dims)
        self.fc_logvar = nn.Linear(64, latent_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dims, output_dim):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.output_dim = output_dim
        self.fc1 = nn.Linear(latent_dims, 64)
        self.bn1 = nn.BatchNorm1d(64)  # 批归一化层
        self.fc2 = nn.Linear(64, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)  # 批归一化层

    def forward(self, z):
        z = F.relu(self.bn1(self.fc1(z)))
        x_recon = torch.sigmoid(self.bn2(self.fc2(z)))
        return x_recon


def norm_layer(channels, norm_type='gn'):
    if norm_type == 'bn':
        return nn.BatchNorm2d(channels)
    elif norm_type == 'gn':
        return nn.GroupNorm(16, channels)
    elif norm_type == 'gn2':
        return nn.GroupNorm(2, channels)
    elif norm_type == 'gn4':
        return nn.GroupNorm(4, channels)
    elif norm_type == 'gn8':
        return nn.GroupNorm(8, channels)
    elif norm_type == 'gn32':
        return nn.GroupNorm(32, channels)
    elif norm_type == 'in':
        return nn.InstanceNorm2d(channels)


class lenet(nn.Module):
    def __init__(self, norm_type=None, in_channel=3):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 2)  # 2=num_classes

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# NOTE latent dims usualy 20
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         c = 64
#         latent_dims = 20
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=c,
#                                kernel_size=4, stride=2, padding=1)  # out: c x 14 x 14
#         self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2,
#                                kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7
#         self.fc_mu = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
#         self.fc_logvar = nn.Linear(
#             in_features=c*2*7*7, out_features=latent_dims)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         # flatten batch of multi-channel feature maps to a batch of feature vectors
#         x = x.view(x.size(0), -1)
#         x_mu = self.fc_mu(x)
#         x_logvar = self.fc_logvar(x)
#         return x_mu, x_logvar


# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         c = 64
#         latent_dims = 20
#         self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*7*7)
#         self.conv2 = nn.ConvTranspose2d(
#             in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
#         self.conv1 = nn.ConvTranspose2d(
#             in_channels=c, out_channels=3, kernel_size=4, stride=2, padding=1)

#     def forward(self, x):
#         capacity = 64
#         x = self.fc(x)
#         # unflatten batch of feature vectors to a batch of multi-channel feature maps
#         x = x.view(x.size(0), capacity*2, 7, 7)
#         x = F.relu(self.conv2(x))
#         # last layer before output is sigmoid, since we are using BCE as reconstruction loss
#         x = torch.sigmoid(self.conv1(x))
#         return x


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar, sampleme=False):
        if self.training or sampleme:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def vae():
    return VariationalAutoencoder()


class TestVAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(TestVAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*4*4)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def latent_sample(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar


def testvae():
    return TestVAE(nc=3, ngf=128, ndf=128, latent_variable_size=500)


class BigEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim):
        super(BigEncoder, self).__init__()
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1]*4, latent_dim)

    def forward(self, x):
        # print(x.shape) [BS, 3, 256, 256]
        x = self.encoder(x)
        x = torch.flatten(x, 1)  # flatten everything except batch
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class BigDecoder(nn.Module):
    def __init__(self, hidden_dims, latent_dim):
        super(BigDecoder, self).__init__()
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.fc = nn.Linear(latent_dim, hidden_dims[-1] * 4)  # decoder input
        hidden_dims.reverse()
        # Build decoder
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()  # NOTE source includes this
        )

    def forward(self, x):
        result = self.fc(x)  # latent into network
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

# from https://github.com/AntixK/PyTorch-VAE/tree/master CITED


class BigVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims, **kwargs):
        super(BigVAE, self).__init__()

        hidden_dims = [32, 64, 128, 256, 512]
        self.encoder = BigEncoder(
            in_channels, hidden_dims=hidden_dims, latent_dim=latent_dim)
        self.decoder = BigDecoder(
            hidden_dims=hidden_dims, latent_dim=latent_dim)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar, sampleme=False):
        if self.training or sampleme:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # return eps * std + mu


def celebavae():
    hidden_dims = [32, 64, 128, 256, 512]
    return BigVAE(in_channels=3, latent_dim=128, hidden_dims=hidden_dims)


# from https://github.com/shivakanthsujit/VAE-PyTorch/blob/master/DCVAE.py CITED
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 1024, 1, 1)


class DCVAEEncoder(nn.Module):
    def __init__(self, image_channels, hidden_size, latent_size):
        super(DCVAEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2),
            nn.LeakyReLU(0.2),
            Flatten(),
        )
        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        return mean, log_var


class DCVAEDecoder(nn.Module):
    def __init__(self, image_channels, hidden_size, latent_size):
        super(DCVAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(hidden_size, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 6, 2),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(latent_size, hidden_size)

    def forward(self, z):
        x = self.fc(z)
        x = self.decoder(x)
        return x


class DCVAE(nn.Module):
    def __init__(
        self,
        image_channels=3,
        image_dim=32,
        hidden_size=1024,
        latent_size=32,
    ):
        super(DCVAE, self).__init__()
        self.encoder = DCVAEEncoder(image_channels, hidden_size, latent_size)
        self.decoder = DCVAEDecoder(image_channels, hidden_size, latent_size)

    def latent_sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x_mu, x_logvar = self.encoder(x)
        z = self.latent_sample(x_mu, x_logvar)
        x_recon = self.decoder(z)
        return x_recon, x_mu, x_logvar


def svhnvae():
    return DCVAE(image_channels=3, image_dim=32, hidden_size=1024, latent_size=32)


# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''


# __all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', norm_type='bn'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes, norm_type=norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes, norm_type=norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    norm_layer(self.expansion * planes, norm_type=norm_type)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_type='bn'):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(16, norm_type=norm_type)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1, norm_type=norm_type)
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2, norm_type=norm_type)
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2, norm_type=norm_type)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, norm_type):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                          stride, norm_type=norm_type))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def num_rep(self):
        return 3

    def representation(self, x, ind=4, to_detach=False):
        bs = x.shape[0]
        res = []
        out = F.relu(self.bn1(self.conv1(x)))
        # res.append(out.detach().reshape([bs, -1]) if to_detach else out.reshape([bs, -1]))
        # if ind == 0:
        #     return res
        out = self.layer1(out)
        res.append(out.detach().reshape(
            [bs, -1]) if to_detach else out.reshape([bs, -1]))
        if ind == 1:
            return res
        out = self.layer2(out)
        res.append(out.detach().reshape(
            [bs, -1]) if to_detach else out.reshape([bs, -1]))
        if ind == 2:
            return res
        out = self.layer3(out)
        res.append(out.detach().reshape(
            [bs, -1]) if to_detach else out.reshape([bs, -1]))
        if ind == 3:
            return res
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        res.append(out.detach().reshape(
            [bs, -1]) if to_detach else out.reshape([bs, -1]))
        return res

    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if return_feature:
            return out, feature
        else:
            return out


def resnet20(norm_type='bn'):
    return ResNet(BasicBlock, [3, 3, 3], norm_type=norm_type, num_classes=2)


def resnet32(norm_type='bn'):
    return ResNet(BasicBlock, [5, 5, 5], norm_type=norm_type, num_classes=2)


def resnet44(norm_type='bn'):
    return ResNet(BasicBlock, [7, 7, 7], norm_type=norm_type, num_classes=2)


def resnet56(norm_type='bn'):
    return ResNet(BasicBlock, [9, 9, 9], norm_type=norm_type, num_classes=2)


def resnet110(norm_type='bn'):
    return ResNet(BasicBlock, [18, 18, 18], norm_type=norm_type, num_classes=2)


def resnet1202(norm_type='bn'):
    return ResNet(BasicBlock, [200, 200, 200], norm_type=norm_type)
