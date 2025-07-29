import torch.nn as nn

def get_activation_function(relu_slope):
    if relu_slope:
        return nn.LeakyReLU(relu_slope)
    else:
        return nn.ReLU()

def get_linear_block(dim_in, dim_out, batch_norm, relu_slope, dopout):
    liniear_block = []
    liniear_block.append(
            nn.Linear(dim_in, dim_out)
    )
    
    if batch_norm:
        liniear_block.append(nn.BatchNorm1d(dim_out))
    
    liniear_block.append(get_activation_function(relu_slope))

    if dopout:
        liniear_block.append(nn.Dropout1d(dopout))

    return liniear_block

def get_conv_block(in_channels, out_channels, batch_norm, relu_slope, dopout):
    conv_block = []

    conv_block.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            stride=2,
            kernel_size=3,
            bias=False,
            padding=0,
        )
    )
    
    if batch_norm:
        conv_block.append(nn.BatchNorm2d(out_channels))
    
    conv_block.append(get_activation_function(relu_slope))

    if dopout:
        conv_block.append(nn.Dropout2d(dopout))

    return conv_block

def get_deconv_block(in_channels, out_channels, batch_norm, relu_slope, dopout):
    deconv_block = []
    deconv_block.append(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            stride=2,
            kernel_size=4,
            bias=False,
            padding=0
        )
    )
    
    if batch_norm:
        deconv_block.append(nn.BatchNorm2d(out_channels))
    
    deconv_block.append(get_activation_function(relu_slope))

    if dopout:
        deconv_block.append(nn.Dropout2d(dopout))

    return deconv_block
