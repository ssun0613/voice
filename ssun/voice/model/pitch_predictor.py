import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True):
        super(Conv_layer, self).__init__()
        self.Conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.Conv_layer(x)
        x = x.contiguous().transpose(1, 2)
        return x

class VariancePredictor(nn.Module):
    def __init__(self):
        super(VariancePredictor, self).__init__()
        self.VariancePredictor = nn.Sequential( Conv_layer(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
                                                nn.ReLU(),
                                                nn.LayerNorm(256),
                                                nn.Dropout(0.5),
                                                Conv_layer(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
                                                nn.ReLU(),
                                                nn.LayerNorm(256),
                                                nn.Dropout(0.5),
                                                nn.Linear(256, 1))

    def set_input(self, encoder_output):
        self.input = encoder_output

    def forward(self):
        out = VariancePredictor(self.input)
        out = out.squeeze(-1)
        return out
