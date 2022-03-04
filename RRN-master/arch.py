##
# @file arch.py
# @brief This is the file where the network architecture lives. This was, except from documentation, which was very lacking in the original code, nearly untouched.

## Imports
from __future__ import absolute_import
from torch.nn import functional as F
from torch.nn import init
import torch.nn as nn
import functools
import torch

def make_layer(block, n_layers):
    """! This concatenates Residual Blocks n_layers times
    @param block  This is the residual Block that will be concatenated
    @param n_layers  This is the count of residual blocks that the network will consist of
    """
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):
    """! Here is the structure of the residual blocks defined
    """
    def __init__(self, nf=64):
        """! Here is the initialization sequence of the Residual Blocks
        """
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) 
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        """! This is the forward pass of one residual block, where the variable identity is the output of the previous block, which gets added as identity skip on return
        @return  Returns the computed output added to the previouos blocks output
        """
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out   

class neuro(nn.Module):
    """! Core of the network. Here the input layer, the residual blocks and the last hidden and output layer are concatenated and the weights are initialized
    """
    def __init__(self, n_c, n_b, scale):
        """! Initializes the whole network
        @param n_c  Sets the count of neurons per layer
        @param n_b  Sets the count of blocks used
        @param scale  Sets the super resolution scale
        """
        super(neuro,self).__init__()
        pad = (1,1)
        ## This is the input layer of the network
        self.conv_1 = nn.Conv2d(scale**2*3 + n_c + 3*2, n_c, (3,3), stride=(1,1), padding=pad) 
        basic_block = functools.partial(ResidualBlock_noBN, nf=n_c)
        ## This are all the residual blocks of the neural network truncated
        self.recon_trunk = make_layer(basic_block, n_b)     # Chains residual blocks with identity skip connections between each block
        ## This is the hidden state
        self.conv_h = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=pad)
        ## This is the output layer
        self.conv_o = nn.Conv2d(n_c, scale**2*3, (3,3), stride=(1,1), padding=pad)
        initialize_weights([self.conv_1, self.conv_h, self.conv_o], 0.1)
    def forward(self, x, hh, o):
        """! Defines the forward pass through the network
        @param x  The current and the previous lowres frame
        @param hh  The previous hidden state 
        @param o  The previous upscaled prediction
        @return  Returns hidden state and output
        """
        x = torch.cat((x, hh, o), dim=1)
        x = F.relu(self.conv_1(x))
        x = self.recon_trunk(x)
        x_h = F.relu(self.conv_h(x))
        x_o = self.conv_o(x)
        return x_h, x_o

class RRN(nn.Module):
    """! This clas is called from our main methods and provides the interface to integrate our network
    """
    def __init__(self, scale, n_c, n_b):
        """! When initialized the whone network is generated and setup for further use
        @param scale  This is the super resolution scale that is used
        @param n_c  This is the neuron count if one block
        @param n_b  This is the count of total blocks which are initialized 
        """
        super(RRN, self).__init__()
        ## Core of the model, initializes the core part of the model
        self.neuro = neuro(n_c, n_b, scale)
        ## Scale of by which the frame is to be upscaled
        self.scale = scale
        ## Function to downsample a frame by scale
        self.down = PixelUnShuffle(scale) 
        self.n_c = n_c
        
    def forward(self, x, x_h, x_o, init):
        """! This is the foward pass through the network, the two low res frames are split and then mended again on another dimension. Then the new hr frame is calculated and added onto the previous frame
        @param x  This is the current and previous lowres frames
        @param x_h  This is the hidden state of the previous frame
        @param x_o  This is the high resolution frame of the previous pass through
        @param init  Is set if its the first frame of a pass through
        @return The hidden state and the predicted hr frame are returned
        """
        _,_,T,_,_ = x.shape
        f1 = x[:,:,0,:,:]
        f2 = x[:,:,1,:,:]
        x_input = torch.cat((f1, f2), dim=1)
        if init:
            x_h, x_o = self.neuro(x_input, x_h, x_o)
        else:
            x_o = self.down(x_o)
            x_h, x_o = self.neuro(x_input, x_h, x_o)
        x_o = F.pixel_shuffle(x_o, self.scale) + F.interpolate(f2, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x_h, x_o

def pixel_unshuffle(input, upscale_factor):
    """! downsizes the input by the upscale_factur
    @param input  Frame to be downsized
    @param upscale_factor  Factor by which the frame should be downsized
    @return downscaled frame
    """
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):
    """! Overwrites the torch.nn.PixelUnshuffle class
    """
    def __init__(self, upscale_factor):
        """!
        @param upscale_factor  Factor by which the frame should be downscaled by
        """
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor
    def forward(self, input):
        """!
        Overwrites the forward pass with the function above
        @param input  Frame that gets downscaled
        @return  Returns the downscaled frame
        """
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

def initialize_weights(net_l, scale=0.1):
    """! Initializes all passed conv layers 
    @param net_l  List of conv layers that should be initialized
    @param scale  float which all weights are set to
    """
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
                m.float()       # Normalization layers have to be float because of convergence issues
            m
