from collections import OrderedDict
from torch import nn
from utils.data import classes

import torch
import torch.nn.functional as F
import numpy as np

display = False


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.manual_seed(0)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    """
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        Adopted Pytorch implementation: https://github.com/sksq96/pytorch-vae
    """
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        # super(VAE, self).__init__()
        # Refer to this post: https://discuss.pytorch.org/t/what-do-i-need-to-inherit-to-make-a-custom-nn-module/5896/2
        super(type(self), self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.register_buffer('scene_dist', torch.zeros((len(classes), 2, z_dim)))
        self.register_buffer('eps', torch.randn(z_dim))

        # Sequential NN to handle Classification Task
        self.classifier = nn.Sequential(OrderedDict([
            ('cls_fc1', nn.Linear(z_dim, len(classes)))
            # ('relu', nn.ReLU()),
            # ('cls_fc2', nn.Linear(31, 30)),
            # ('activation', nn.LogSoftmax(dim=1)) # CrossEntropy applies logSoftmax expects fc
        ]))

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    # Reparameterize based on Scene distribution
    # TODO Does not generate scene image for comparison
    def reparameterize(self, mu, logvar, labels):
        stds = logvar.mul(0.5).exp_()

        for label, mean, std in zip(labels, mu, stds):
            # Add the current image with the corresponding scene distribution
            # if label in scene_dist:
            # Concatenate Mean and Std to Scene Distribution
            label = label.item()
            self.scene_dist[label][0] += mean
            self.scene_dist[label][1] += std

            # Image distribution information
            if display:
                print("Scene dist_shape", self.scene_dist[label].shape)
                print(self.scene_dist[label])
                print(f'{classes[label]} \t| {mean.type()} \t| {std.type()}')

        # Caluclate latent space for each scene
        z = mu + stds * self.eps
        return z

    def bottleneck(self, h, label):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar, label)
        return z, mu, logvar

    def encode(self, x, label):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h, label)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x, label):
        z, mu, logvar = self.encode(x, label)
        class_pred = self.classifier(z)
        z = self.decode(z)
        return z, mu, logvar, class_pred



class VAE_BN(nn.Module):
    """
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 + Batch Normalization.
        Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. 2015
    """
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        # super(VAE, self).__init__()
        # Refer to this post: https://discuss.pytorch.org/t/what-do-i-need-to-inherit-to-make-a-custom-nn-module/5896/2
        super(type(self), self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Flatten()
        )


        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.register_buffer('scene_dist', torch.zeros((len(classes), 2, z_dim)))
        self.register_buffer('eps', torch.randn(z_dim))

        # Sequential NN to handle Classification Task
        self.classifier = nn.Sequential(OrderedDict([
            ('cls_fc1', nn.Linear(z_dim, len(classes)))
            # ('relu', nn.ReLU()),
            # ('cls_fc2', nn.Linear(31, 30)),
            # ('activation', nn.LogSoftmax(dim=1)) # CrossEntropy applies logSoftmax expects fc
        ]))

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    # Reparameterize based on Scene distribution
    # TODO Does not generate scene image for comparison
    def reparameterize(self, mu, logvar, labels):
        stds = logvar.mul(0.5).exp_()

        for label, mean, std in zip(labels, mu, stds):
            # Add the current image with the corresponding scene distribution
            # if label in scene_dist:
            # Concatenate Mean and Std to Scene Distribution
            label = label.item()
            self.scene_dist[label][0] += mean
            self.scene_dist[label][1] += std

            # Image distribution information
            if display:
                print("Scene dist_shape", self.scene_dist[label].shape)
                print(self.scene_dist[label])
                print(f'{classes[label]} \t| {mean.type()} \t| {std.type()}')

        # Caluclate latent space for each scene
        z = mu + stds * self.eps
        return z

    def bottleneck(self, h, label):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar, label)
        return z, mu, logvar

    def encode(self, x, label):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h, label)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x, label):
        z, mu, logvar = self.encode(x, label)
        class_pred = self.classifier(z)
        z = self.decode(z)
        return z, mu, logvar, class_pred

class VAE_GN(nn.Module):
    """
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 + Batch Normalization.
        Wu and He. Group normalization. 2018
    """
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        # super(VAE, self).__init__()
        # Refer to this post: https://discuss.pytorch.org/t/what-do-i-need-to-inherit-to-make-a-custom-nn-module/5896/2
        super(type(self), self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.GroupNorm(64, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.GroupNorm(128, 256),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.register_buffer('scene_dist', torch.zeros((len(classes), 2, z_dim)))
        self.register_buffer('eps', torch.randn(z_dim))

        # Sequential NN to handle Classification Task
        self.classifier = nn.Sequential(OrderedDict([
            ('cls_fc1', nn.Linear(z_dim, len(classes)))
            # ('relu', nn.ReLU()),
            # ('cls_fc2', nn.Linear(31, 30)),
            # ('activation', nn.LogSoftmax(dim=1)) # CrossEntropy applies logSoftmax expects fc
        ]))

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.GroupNorm(64, 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.GroupNorm(1, 3),
            nn.Sigmoid(),
        )

    # Reparameterize based on Scene distribution
    # TODO Does not generate scene image for comparison
    def reparameterize(self, mu, logvar, labels):
        stds = logvar.mul(0.5).exp_()

        for label, mean, std in zip(labels, mu, stds):
            # Add the current image with the corresponding scene distribution
            # if label in scene_dist:
            # Concatenate Mean and Std to Scene Distribution
            label = label.item()
            self.scene_dist[label][0] += mean
            self.scene_dist[label][1] += std

            # Image distribution information
            if display:
                print("Scene dist_shape", self.scene_dist[label].shape)
                print(self.scene_dist[label])
                print(f'{classes[label]} \t| {mean.type()} \t| {std.type()}')

        # Caluclate latent space for each scene
        z = mu + stds * self.eps
        return z

    def bottleneck(self, h, label):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar, label)
        return z, mu, logvar

    def encode(self, x, label):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h, label)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x, label):
        z, mu, logvar = self.encode(x, label)
        class_pred = self.classifier(z)
        z = self.decode(z)
        return z, mu, logvar, class_pred
# Reconstruction + KL divergence losses summed over all elements and batch
def SE_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def compare(x):
    recon_x, _, _ = VAE(x)
    return torch.cat([x, recon_x])

