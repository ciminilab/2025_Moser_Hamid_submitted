import torch.nn as nn

from .utils.model_blocks import get_linear_block
from .vae_base import VAE

class VAE_LINEAR(VAE):
    def __init__(self, model_config) -> None:
        hidden_dims = model_config.hidden_dims
        super().__init__(h_dim=hidden_dims[-1], z_dim=model_config.z_dim)

        # Build Encoder
        dim_in = model_config.input_dim
        encoder_modules = []
        for dim_out in hidden_dims:
            encoder_modules.extend(
                get_linear_block(dim_in, dim_out, model_config.batch_norm, model_config.relu_slope, model_config.dropout)
            )
            dim_in = dim_out

        self.encoder = nn.Sequential(
            *encoder_modules
        )

        # Build decoder
        decoder_modules = []
        hidden_dims.reverse()
        dim_in = model_config.z_dim
        for dim_out in hidden_dims:
            decoder_modules.extend(
                get_linear_block(
                    dim_in, dim_out,
                    model_config.batch_norm,
                    model_config.relu_slope,
                    model_config.dropout
                )
            )
            dim_in = dim_out

        self.decoder = nn.Sequential(
            *decoder_modules,
            *get_linear_block(
                    dim_out,
                    model_config.input_dim,
                    model_config.batch_norm,
                    model_config.relu_slope,
                    model_config.dropout
                ),
        )
