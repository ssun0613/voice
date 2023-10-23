import numpy as np
import torch
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class Discriminator(nn.Module):
    """PatchGAN discriminator"""
    def __init__(self, opt, device):
        super(Discriminator, self).__init__()
        residual_in_channels = opt.n_bins
        self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=residual_in_channels // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        GLU())
        # Downsampling Layers
        self.downSample_layer = nn.Sequential(self.downsample(in_channels=residual_in_channels // 2, out_channels=residual_in_channels, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                              self.downsample(in_channels=residual_in_channels, out_channels=residual_in_channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                              self.downsample(in_channels=residual_in_channels * 2, out_channels=residual_in_channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1))

        self.downSample4 = self.downsample(in_channels=residual_in_channels * 4, out_channels=residual_in_channels * 4,kernel_size=(1, 10), stride=(1, 1), padding=(0, 2))
        # Conv Layer
        self.outputConvLayer = nn.Conv2d(in_channels=residual_in_channels * 4, out_channels=1,kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

    def downsample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                  nn.InstanceNorm2d(num_features=out_channels, affine=True),
                                  GLU())
        return convLayer

    def forward(self, x):
        # x has shape [batch_size, num_features, frames]
        # discriminator requires shape [batchSize, 1, num_features, frames]
        # x has shape [10, 192, 80]
        # discriminator requires shape [10, 1, 192, 80]

        # conv_layer_1.shape : torch.Size([10, 128, 192, 80])
        # downsample.shape : torch.Size([10, 1024, 24, 10])
        # output.shape : torch.Size([10, 1, 24, 10])

        x = x.unsqueeze(1)
        conv_layer_1 = self.convLayer1(x)
        downsample = self.downSample_layer(conv_layer_1)
        output = torch.sigmoid(self.outputConvLayer(downsample))
        return output

if __name__ == '__main__':
    from ssun.Voice_trans.config import Config
    config = Config()
    device = torch.device("cpu")

    np.random.seed(0)
    input = torch.from_numpy(np.random.randn(10, 192, 80)).float()
    discriminator = Discriminator(config.opt, device)
    output = discriminator(input)
    print("Discriminator output shape ", output.shape)