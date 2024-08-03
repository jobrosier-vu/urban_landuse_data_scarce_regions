# -*- coding: utf-8 -*-
"""
 code for densenet from: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html#DenseNet
 
 spatial pyramid pooling model structure from: J. Arndt and D. Lunga, "Large-Scale Classification of Urban Structural Units From Remote Sensing Imagery," 
 in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 2634-2648, 2021, 
 doi: 10.1109/JSTARS.2021.3052961. 

@author: JJR226
"""

import torch.nn as nn
import torch
from types import SimpleNamespace


act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}

class spatial_pyramid_pool_adaptive(nn.Module):
    def __init__(self, bin_size):
        '''
        bin_size: a int vector of bin_sizes
        
        returns: a tensor vector with shape [m x n] is the concentration of multi-level pooling
        '''    
        super().__init__()
        self.maxpool1 = nn.AdaptiveMaxPool2d((bin_size[0],bin_size[0]))
        self.maxpool2 = nn.AdaptiveMaxPool2d((bin_size[1],bin_size[1]))
        self.maxpool3 = nn.AdaptiveMaxPool2d((bin_size[2],bin_size[2])) 
        
    def forward(self, x):
        
        out1 = torch.flatten(self.maxpool1(x),start_dim = 1)
        out2 = torch.flatten(self.maxpool2(x),start_dim = 1)
        out3 = torch.flatten(self.maxpool3(x),start_dim = 1)
        
        spp = torch.cat([out1,out2,out3], 1)
        return spp
            

class spatial_pyramid_pool(nn.Module):
    def __init__(self, bin_size):
        '''
        bin_size: a int vector of bin_sizes
        
        returns: a tensor vector with shape [m x n] is the concentration of multi-level pooling
        '''    
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=bin_size[0],stride=bin_size[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=bin_size[1],stride=bin_size[1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=bin_size[2],stride=bin_size[2])
        
    def forward(self, x):
        
        out1 = torch.flatten(self.maxpool1(x),start_dim = 1)
        out2 = torch.flatten(self.maxpool2(x),start_dim = 1)
        out3 = torch.flatten(self.maxpool3(x),start_dim = 1)
        
        spp = torch.cat([out1,out2,out3], 1)
        return spp

class DenseLayer(nn.Module):
    def __init__(self, c_in, bn_size, growth_rate, act_fn):
        """
        Inputs:
            c_in - Number of input channels
            bn_size - Bottleneck size (factor of growth rate) for the output of the 1x1 convolution. Typically between 2 and 4.
            growth_rate - Number of output channels of the 3x3 convolution
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            act_fn(),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.cat([out, x], dim=1)
        return out
    
    
class DenseBlock(nn.Module):
    def __init__(self, c_in, num_layers, bn_size, growth_rate, act_fn):
        """
        Inputs:
            c_in - Number of input channels
            num_layers - Number of dense layers to apply in the block
            bn_size - Bottleneck size to use in the dense layers
            growth_rate - Growth rate to use in the dense layers
            act_fn - Activation function to use in the dense layers
        """
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            # Input channels are original plus the feature maps from previous layers
            layer_c_in = c_in + layer_idx * growth_rate
            layers.append(DenseLayer(c_in=layer_c_in, bn_size=bn_size, growth_rate=growth_rate, act_fn=act_fn))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out    
    
    
class TransitionLayer(nn.Module):
    def __init__(self, c_in, c_out, act_fn):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Average the output for each 2x2 pixel group
        )

    def forward(self, x):
        return self.transition(x)    
    
class TransitionLayer_last(nn.Module):
    def __init__(self, c_in, c_out, act_fn):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.transition(x) 
        
class Building_footprint_part(nn.Module):
    def __init__(self,in_channels_BF,out_channels, *args, **kwargs):
        super().__init__()
        self.Lin1 = nn.Linear(in_channels_BF, 256)
        self.Lin2 = nn.Linear(256, 256)
        self.Lin3 = nn.Linear(256, out_channels)
        self.BN = nn.BatchNorm1d(num_features=256)
        self.Rel = nn.ReLU()
        

    def forward(self, x):
        x = self.Lin1(x)
        x = self.BN(self.Rel(x))
        x = self.Lin2(x)
        x = self.Rel(x)  
        x = self.Lin2(x) 
        x = self.Rel(x) 
        out = self.Lin3(x)
        
        return out 
       
    
class DenseNet_BF(nn.Module):
    def __init__(
        self, num_input_layers = 3,in_channels_BF=13,num_classes=10, num_layers=[6, 12], bn_size=2, growth_rate=32, act_fn_name="relu",lin_size=3584,bin_size=[1,2,4], **kwargs
    ):
        super().__init__()
        self.hparams = SimpleNamespace(
            num_input_layers=num_input_layers,
            num_classes=num_classes,
            num_layers=num_layers,
            bn_size=bn_size,
            growth_rate=growth_rate,
            act_fn_name=act_fn_name,
            act_fn=act_fn_by_name[act_fn_name],
            bin_size=bin_size,
            lin_size = lin_size,

        )
        self.building_footprint_part = Building_footprint_part(in_channels_BF,num_classes)
        self._create_network()
        self._init_params()
     
    
    def _create_network(self):
         c_hidden = self.hparams.growth_rate * self.hparams.bn_size  # The start number of hidden channels
    
         # A first convolution on the original image to scale up the channel size
         self.input_net = nn.Sequential(
             # No batch norm or activation function as done inside the Dense layers
             nn.Conv2d(self.hparams.num_input_layers, c_hidden, kernel_size=7,stride=2, padding=3),
             nn.BatchNorm2d(c_hidden),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         )
    
         # Creating the dense blocks, eventually including transition layers
         blocks = []
         for block_idx, num_layers in enumerate(self.hparams.num_layers):
             blocks.append(
                 DenseBlock(
                     c_in=c_hidden,
                     num_layers=num_layers,
                     bn_size=self.hparams.bn_size,
                     growth_rate=self.hparams.growth_rate,
                     act_fn=self.hparams.act_fn,
                 )
             )
             c_hidden = c_hidden + num_layers * self.hparams.growth_rate  # Overall output of the dense block
             # apply transition layer
    
    
             if block_idx < len(self.hparams.num_layers) - 1:  # Don't apply transition layer on last block
                 blocks.append(TransitionLayer(c_in=c_hidden, c_out=c_hidden // 2, act_fn=self.hparams.act_fn))
                 c_hidden = c_hidden // 2
             else:
                 blocks.append(TransitionLayer_last(c_in=c_hidden, c_out=c_hidden // 2, act_fn=self.hparams.act_fn))
                 c_hidden = c_hidden // 2
    
         self.blocks = nn.Sequential(*blocks)
    
    
                
         # Mapping to classification output
         self.output_net = nn.Sequential(
             nn.BatchNorm2d(c_hidden), # The features have not passed a non-linearity until here.
             self.hparams.act_fn(),
             spatial_pyramid_pool(self.hparams.bin_size), 
             nn.Linear(self.hparams.lin_size,256),
             nn.Dropout(p=0.5),
             #nn.LazyLinear(1024),
             nn.ReLU(inplace=True),
             nn.Linear(256, 128),
             nn.Dropout(p=0.3),
             nn.ReLU(inplace=True),
             nn.Linear(128, self.hparams.num_classes),
         )
    
    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the
        # convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
       
    def forward(self, z):
        x = self.input_net(z[0])
        x = self.blocks(x)
        x = self.output_net(x)
        b1 = self.building_footprint_part(z[1])
        b2 = self.building_footprint_part(z[2])
        return x,b1,b2
    
    
if __name__ == "__main__":
    from torchsummary import summary
    model = DenseNet_BF(num_input_layers = 3,in_channels_BF = 13,num_classes=5, num_layers=[6, 12], bn_size=2, growth_rate=32,lin_size=5376, act_fn_name="relu",bin_size=[31,15,7]).cuda()
    summary(model, input_size=[(1,3, 251, 251)])