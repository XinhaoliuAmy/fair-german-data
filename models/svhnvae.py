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
