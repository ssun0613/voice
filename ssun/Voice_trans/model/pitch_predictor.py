import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True):
        super(Conv_layer, self).__init__()
        self.Conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1,2).to(x.device).float()
        out = self.Conv_layer(x)
        out = out.contiguous().transpose(1, 2)

        return out

class pitch_predictor(nn.Module):
    def __init__(self):
        super(pitch_predictor, self).__init__()
        self.pitch_predicton = nn.LSTM(input_size=24, hidden_size=32, num_layers=8, batch_first=True, bidirectional=True)

    def forward(self, r_c_s):
        r_c_s = torch.tensor(r_c_s, dtype=torch.float32)
        p = self.pitch_predicton(r_c_s)[0]

        p_forward = p[:, :, :32]
        p_backward = p[:, :, 32:]

        pitch_p = torch.cat((p_forward[:, 7::8, :], p_backward[:, ::8, :]), dim=-1)

        return pitch_p


if __name__ == '__main__':
    model = pitch_predictor()
    x = torch.rand(2, 192, 24)
    pitch_p = model.forward(x)
