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

class pitch_predictor(nn.Module):
    def __init__(self):
        super(pitch_predictor, self).__init__()
        self.pitch_predictor = nn.Sequential( Conv_layer(in_channels = 24, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                                              nn.ReLU(),
                                              nn.LayerNorm(128),
                                              nn.Dropout(0.5),
                                              Conv_layer(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
                                              nn.ReLU(),
                                              nn.LayerNorm(256),
                                              nn.Dropout(0.5),
                                              nn.Linear(256, 1))

        self.pitch_embedding = nn.Embedding(256, 256)

    def forward(self, r_c_s):
        out = self.pitch_predictor(r_c_s)
        out = out.squeeze(-1)

        prediction = out * 1.0
        embedding = self.pitch_embedding(torch.bucketize(prediction, self.pitch_bins))

        return prediction, embedding

if __name__ == '__main__':
    model = pitch_predictor()
    x = torch.rand(2, 192, 24)
    model.forward(x)
