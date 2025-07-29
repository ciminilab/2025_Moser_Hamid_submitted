import torch
import torch.nn as nn

# Base class that implements a Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, h_dim, z_dim) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.z_mean = torch.nn.Linear(h_dim, z_dim)
        self.z_sigma = torch.nn.Linear(h_dim, z_dim)
        

    def construct_z(self, z_mean, z_sigma):
        epsilon = torch.randn_like(z_sigma)
        z = z_mean + epsilon * torch.exp(0.5 * z_sigma) 
        return z

    def forward(self, x):
        x_encoded = self.encoder(x)

        z_mean, z_sigma = self.z_mean(x_encoded), self.z_sigma(x_encoded)
        z_vec = self.construct_z(z_mean, z_sigma)

        x_recontsructed = self.decoder(z_vec)
        return x_recontsructed, z_mean, z_sigma

    def test_shape(self, example_input):
        print(f'Shape of input x: {example_input.shape}')

        x_encoded = self.encoder(example_input)
        print(f'Shape of encoded x: {x_encoded.shape}')
        assert x_encoded.shape[-1] == self.h_dim, f'Invalid shape of hidden layer! Encoder output is of length {x_encoded.shape[1]} but should be {self.h_dim}'

        z_mean, z_sigma = self.z_mean(x_encoded), self.z_sigma(x_encoded)
        z_vec = self.construct_z(z_mean=z_mean, z_sigma=z_sigma)
        assert z_vec.shape[-1] == self.z_dim, f'Invalid shape of latent vector: z is of length {z_vec.shape[1]} but should be {self.z_dim}'
        print(f'Shape of latent vector z: {z_vec.shape}')

        x_recontsructed = self.decoder(z_vec)
        assert x_recontsructed.shape == example_input.shape, f'Invalid shape of reconstructed vector: x_reconstructed is of length {x_recontsructed.shape} but should be {example_input.shape}'
        print(f'Shape of reconstructed x: {x_recontsructed.shape}')
