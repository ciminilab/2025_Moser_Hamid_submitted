import torch
import torch.nn as nn

from .utils.model_blocks import get_conv_block, get_deconv_block
from .utils.helper_functions import Reshape, Trim
from .vae_base import VAE
from .model_config import CNNVAEConfig

# Convolutional Autoencoder
class VAE_CNN(VAE):
    def __init__(self, model_config: CNNVAEConfig) -> None:
        input_shape = model_config.input_shape

        # Build the encoder
        hidden_dims = model_config.hidden_dims
        encoder_modules = []
        for i in range(len(hidden_dims)-1):
            encoder_modules.extend(
                get_conv_block(hidden_dims[i], hidden_dims[i+1], model_config.batch_norm, model_config.relu_slope, model_config.dropout)
            )

        # Caluculate the output dimension of the last encoder layer
        hidden_layer_dimension, flattened_length = self._get_hidden_dimension(encoder_modules, 32, input_shape)

        super().__init__(h_dim=flattened_length, z_dim=model_config.z_dim)

        self.encoder = nn.Sequential(*encoder_modules, nn.Flatten())

        # Build the decoder
        decoder_modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            decoder_modules.extend(
                get_deconv_block(
                    hidden_dims[i],
                    hidden_dims[i+1],
                    model_config.batch_norm,
                    model_config.relu_slope,
                    model_config.dropout
                )
            )

        reconstructed_input_shape = self._get_decoder_shape(model_config.z_dim, flattened_length, decoder_modules, 32, hidden_layer_dimension)
        self.reconstructed_input_shape = reconstructed_input_shape
        self.flattened_length = flattened_length

        self.decoder = nn.Sequential(
            nn.Linear(model_config.z_dim, flattened_length),
            Reshape(hidden_layer_dimension),
            *decoder_modules,
            Trim(input_shape),
        )

        if model_config.apply_sigmoid:
            self.decoder.add_module("sigmoid", nn.Sigmoid())

    def _get_hidden_dimension(self, encoder_modules, batch_size, input_shape):
        encoder = nn.Sequential(*encoder_modules)
        test_tensor = torch.zeros([batch_size, *input_shape])
        hidden = encoder(test_tensor)

        flatten_layer = nn.Sequential(nn.Flatten())
        flattened_hidden = flatten_layer(hidden)

        return list(hidden.shape)[1:], flattened_hidden.shape[-1]

    def _get_decoder_shape(self, z_dim, flattened_length, decoder_modules, batch_size, hidden_shape):
        decoder = self.decoder = nn.Sequential(
            nn.Linear(z_dim, flattened_length),
            Reshape(hidden_shape),
            *decoder_modules
        )

        test_tensor = torch.zeros([batch_size, 1, z_dim])
        reconstructed_input = decoder(test_tensor)

        return reconstructed_input.shape

 
    def test_shape(self, example_input):
        super().test_shape(example_input)
        
        print(f'Length of flattend decoder output: {self.flattened_length}')
        print(f'Shape of reconstructed x (before trimming): {self.reconstructed_input_shape}')
