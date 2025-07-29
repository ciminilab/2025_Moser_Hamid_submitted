import torch.nn as nn
import torch

from .utils.model_blocks import get_conv_block, get_deconv_block
from .utils.helper_functions import Reshape, Trim
from .vae_base import VAE

class DOUBLE_DECODER_VAE_CNN(VAE):
    def __init__(self, model_config) -> None:
        input_shape = model_config.input_shape
        hidden_dims = model_config.hidden_dims

        # Build Encoder
        encoder_modules = []
        for i in range(len(hidden_dims)-1):
            encoder_modules.extend(
                get_conv_block(hidden_dims[i], hidden_dims[i+1], model_config.batch_norm, model_config.relu_slope, model_config.dropout)
            )

        # Calculate the output dimension of the last encoder layer
        hidden_layer_dimension, flattened_length = self._get_hidden_dimension(encoder_modules, 32, input_shape)
        super().__init__(h_dim=flattened_length, z_dim=model_config.z_dim)

        self.encoder = nn.Sequential(*encoder_modules, nn.Flatten())

        ## Build first decoder
        hidden_dims = model_config.decoder_one_hidden_dims
        encoder_modules = []
        for i in range(len(hidden_dims) - 1):
            encoder_modules.extend(
                get_conv_block(
                    hidden_dims[i],
                    hidden_dims[i+1],
                    model_config.batch_norm,
                    model_config.relu_slope,
                    model_config.dropout
                )
            )
        hidden_layer_dimension, flattened_length = self._get_hidden_dimension(encoder_modules, 32, model_config.decoder_one_input_shape)

        # Construct decoder modules
        hidden_dims.reverse()
        decoder_modules = []
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

        self.decoder_one = nn.Sequential(
            nn.Linear(model_config.z_dim, flattened_length),
            Reshape(hidden_layer_dimension),
            *decoder_modules,
            Trim(input_shape),
        )
        if model_config.apply_sigmoid:
            self.decoder_one.add_module("sigmoid", nn.Sigmoid())

        ## Build second decoder
        # Get hidden dimension
        hidden_dims = model_config.decoder_two_hidden_dims
        encoder_modules = []
        for i in range(len(hidden_dims) - 1):
            encoder_modules.extend(
                get_conv_block(
                    hidden_dims[i],
                    hidden_dims[i+1],
                    model_config.batch_norm,
                    model_config.relu_slope,
                    model_config.dropout
                )
            )
        hidden_layer_dimension, flattened_length = self._get_hidden_dimension(encoder_modules, 8, model_config.decoder_two_input_shape)

        # Construct decoder
        hidden_dims.reverse()
        decoder_modules = []
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

        self.decoder_two = nn.Sequential(
            nn.Linear(model_config.z_dim, flattened_length),
            Reshape(hidden_layer_dimension),
            *decoder_modules,
            Trim(input_shape),
        )
        if model_config.apply_sigmoid:
            self.decoder_two.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x_encoded = self.encoder(x)

        z_mean, z_sigma = self.z_mean(x_encoded), self.z_sigma(x_encoded)
        z_vec = self.construct_z(z_mean, z_sigma)

        x_recontsructed_one = self.decoder_one(z_vec)
        x_recontsructed_two = self.decoder_two(z_vec)
        return (x_recontsructed_one, x_recontsructed_two), z_mean, z_sigma

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
