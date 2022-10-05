
import torch
import torch.nn as nn
import numpy as np

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

def get_conv_dim_size(width, kernel_size, padding=0, stride=1):
    return int(((width - kernel_size + 2*padding) / stride) + 1)

class ConvEncoder(nn.Module):

    def __init__(self, input_dims, latent_dim):
        super(ConvEncoder, self).__init__()


        self.enConv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.c1w = get_conv_dim_size(input_dims[0], 3, stride=1)
        self.enConv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.c2w = get_conv_dim_size(self.c1w, 3, stride=2)
        self.enConv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.c3w = get_conv_dim_size(self.c2w, 3, stride=2)
        self.enConv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.c4w = get_conv_dim_size(self.c3w, 3, stride=1)
        self.c4w_shape = [-1, 64, self.c4w, self.c4w]
        self.flatten = nn.Flatten()
        self.flatten_dim = int(self.c4w**2 * 64)
        self.mu = nn.Linear(self.flatten_dim, latent_dim)
        self.log_var = nn.Linear(self.flatten_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        #print("1: ", x.size())
        x = self.LeakyReLU(self.enConv1(x))
        #print("2: ", x.size())
        x = self.LeakyReLU(self.enConv2(x))
        #print("3: ", x.size())
        x = self.LeakyReLU(self.enConv3(x))
        #print("4: ", x.size())
        x = self.LeakyReLU(self.enConv4(x))
        #print("5: ", x.size())
        #print("predicted final image width: {}, flatten dim: {}, x shape: {}".format(self.c4w, self.flatten_dim, x.size()))
        x = self.flatten(x)
        mu = self.mu(x)
        log_var = self.log_var(x)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mu, log_var


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(ConvDecoder, self).__init__()

        self.input_shape = input_shape
        self.expand = nn.Linear(latent_dim, np.prod(input_shape[1:]))
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.expand(x))
        x = torch.reshape(x, self.input_shape)
        #print("1: ", x.size())
        x = self.LeakyReLU(self.deconv1(x))
        #print("2: ", x.size())
        x = self.LeakyReLU(self.deconv2(x))
        #print("3: ", x.size())
        x = self.LeakyReLU(self.deconv3(x))
        #print("4: ", x.size())
        x = torch.sigmoid(self.deconv4(x))
        #print("5: ", x.size())

        return x


class VAE(nn.Module):
    def __init__(self, input_dims, latent_dim):
        super(VAE, self).__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(input_dims, latent_dim)
        final_conv_shape = self.encoder.c4w_shape
        self.decoder = ConvDecoder(latent_dim, final_conv_shape)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var
