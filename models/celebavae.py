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
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
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
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3, kernel_size= 3, padding= 1),
                            nn.Tanh()
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
        self.encoder = BigEncoder(in_channels, hidden_dims=hidden_dims, latent_dim=latent_dim)
        self.decoder = BigDecoder(hidden_dims=hidden_dims, latent_dim=latent_dim)
    
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
