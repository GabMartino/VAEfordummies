from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchinfo import summary

from torch.nn import functional as F
'''
    The VAE (Variational Autoencoder) is made by an Encoder, a Latent space made by
    a Normal distribution and a decoder.
    This tries to regularize the space, that is "Similar features are close to each others"
        
    Theoretical Explaination:
    Let's consider x as an image. This image is sample from a prior probability distribution p(x).
    This p(x) is impossible to estimate for real, since the unlimited possibility of the images.
    
    For this reason we want to approximate OUR own "p(x)" from a set of images that we have!
    
    At first we can "encode" our images in a set of extracted features. For this reason we can use a CNN as an encoder.
    
    Moreover, since our goal is to "RESAMPLE" from the same distribution to recreate similar images, we want to fix a model distribution.
    In general we choose a Gaussian. So we estimate our p(x) with p_theta(x) that is a N(x | mu, var).
    
    Now, how to do this??
    Intuitively if we want that our distribution p_theta(x) is similar to the ipothetical p(x),
    We assume that if we sample from this distribution this will return the same images!!
    
    Thats what the decoder wants to do!
    
    1) Sample from this distribution -> we'll have a feature vector
    2) Decoder this feature vector with a decoder
    3) "See" if it belongs to the "previuous" distribution -> It's a good image.
    
'''

class VAE(pl.LightningModule):
    '''
        :param in_channels: channels of the input image
        :param latent_dim: vector size of the latent space Mean Value, Variance
        :param hidden_dims: List of the size of the features map ( We are considering VAE for images -> CNN)

    '''

    def __init__(self, input_channels: int,
                        latent_space_size: int,
                        hidden_dims: List,
                        kernel_sizes: List,
                        input_size: int,
                        lr: float,
                        kld_weight: float):
        super().__init__()

        self.save_hyperparameters()
        self.latent_dim = latent_space_size
        self.in_channels = input_channels
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.input_size = input_size
        self.lr = lr
        self.kld_weight = kld_weight

        modules = []

        '''
            Encoder
        '''
        self.last_feature_map_size = self.input_size
        in_channels = self.in_channels
        for h_dim, kernel in zip(self.hidden_dims, self.kernel_sizes):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=kernel, stride=2, padding=0),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            self.last_feature_map_size = (self.last_feature_map_size - kernel)/2 + 1

        self.last_feature_map_size = int(self.last_feature_map_size)
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(self.hidden_dims[-1] * (self.last_feature_map_size**2), self.latent_dim)  ## take the last CNN layers and multiply by 4 ??
        self.fc_var = nn.Linear(self.hidden_dims[-1] * (self.last_feature_map_size**2), self.latent_dim)

        #summary(self.encoder, (1, 3, self.input_size, self.input_size))
        # Build Decoder
        modules = []
        #print("init", self.hidden_dims[-1], self.last_feature_map_size)

        '''
            Let's make the first block of the decoder equals to the last of the encoder
        '''
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * (self.last_feature_map_size**2)) ## we map from a single vector to a block for decoder input

        decoder_hidden_dims = self.hidden_dims.copy()
        decoder_hidden_dims.append(decoder_hidden_dims[-1])
        decoder_hidden_dims.reverse()

        decoder_kernel_size = self.kernel_sizes.copy()
        decoder_kernel_size.reverse()

        for i, k in zip(range(len(decoder_hidden_dims) - 1), decoder_kernel_size):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(decoder_hidden_dims[i],
                                       decoder_hidden_dims[i + 1],
                                       kernel_size=k,
                                       stride=2,
                                       padding=0,
                                       output_padding=1),
                    nn.BatchNorm2d(decoder_hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        #summary(self.decoder, (1, self.hidden_dims[-1], self.last_feature_map_size, self.last_feature_map_size))
        self.final_layer = nn.Sequential(nn.Conv2d(decoder_hidden_dims[-1], out_channels=self.in_channels,
                                                    kernel_size=4, padding= 0),
                                        nn.Sigmoid())



    def encode(self, input):
        result = self.encoder(input) ## Encoder
        result = torch.flatten(result, start_dim=1) ## The last feature map is than flatten, may an AvgGlobalMaxPooling whould be better

        '''
            From the encoder we try to define the (mu, log_var)
        '''
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


    '''
        We recall that we are approximating p(x) with N(x| mu, var) ( that is p_theta(x))
        
        This means that we want to sample from a distribution that change mu and var over time (because also theta changes)
        during backprop leading to problems in training.
        For this reason better to sample from a normal distribution N(0,1), than shift in mean and variance with this trick
        
        check: https://gregorygundersen.com/blog/2018/04/29/reparameterization/
    '''
    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):

        result = self.decoder_input(z)## Decode the sampled vector
        result = result.view(-1, self.hidden_dims[-1], self.last_feature_map_size, self.last_feature_map_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    '''
        Sample a certain num_sample images from the model
    '''
    def sample(self, num_sample):

        z = torch.randn(num_sample, self.latent_dim)
        sample = self.decode(z)
        return sample

    def generate(self, z):
        with torch.no_grad():
            return self.decode(z)


    '''
        1) Get the Mu and the Var from the Encoder
        2) Reparameterize from the N(0,1) to N(mu, var) and sample
        3) Decoder the sampled latent vector
    '''
    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var) ## sample
        out = self.decode(z)
        return out, input, z, mu, log_var

    '''
        Now, we recall that we want 2 things!
        
        1) Ability to reconstruct the data: x should be equals to the output x'
        2) The distribution used for sampling should be as close as possible to N(0,1), this 
            helps to regularize the space is such a way similar vectors will correspond to similar image (after decoding)
        
        To better emulate the "loss of a distribution", we can use the KL Divergence KL(Q(x)|| P(x)) that measure how much information we'll lose
        if use use Q(x) instead of P(x). 
        In case of two normal distribution the computation is simpler. Look at https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
        For this reason the loss function could be a combination of two factor: (1) Reconstruction Loss and (2) "Distribution Loss"
        
        
        
    '''
    def loss_function(self, out, input, mu, log_var):

        ## (1) Reconstruction loss
        recontruction_loss = F.mse_loss(out, input)


        ## (2) "Distribution Loss the sum if for the all components of the vectors, and the last mean is for the batch
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recontruction_loss + self.kld_weight*kld_loss


        self.log("Reconstruction_loss", recontruction_loss)
        self.log("KLD_Loss", kld_loss)
        return loss

    '''
    
                    TRAINING METHODS

    '''
    def training_step(self, batch, batch_idx):

        img, labels = batch
        out, input, z, mu, log_var = self.forward(img)
        train_loss = self.loss_function(out, input, mu, log_var)
        self.log("train_loss", train_loss)
        return train_loss

    def training_epoch_end(self, outputs):
        mean_loss = np.mean([ v['loss'].item() for v in outputs])
        self.log("train_loss_epoch", mean_loss)


    def validation_step(self, batch, batch_idx):
        img, labels = batch
        out, input, z, mu, log_var = self.forward(img)
        val_loss = self.loss_function(out, input, mu, log_var)
        self.log("val_loss", val_loss)
        return {"val_loss": val_loss, "z": z, "labels": labels}

    def validation_epoch_end(self, outputs):
        mean_loss = np.mean([ v["val_loss"].item() for v in outputs])
        self.log("val_loss_epoch", mean_loss)



    def predict_step(self, batch, batch_idx):
        img, labels = batch
        out, input, z, mu, log_var = self.forward(img)
        return out, z, labels

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def main():
    pass



if __name__ == "__main__":
    main()