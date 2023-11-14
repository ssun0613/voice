import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class pitch_predictor(nn.Module):
    def __init__(self):
        super(pitch_predictor, self).__init__()
        self.pitch_predicton = nn.LSTM(input_size=24, hidden_size=32, num_layers=8, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(64, 1)

    def forward(self, r_c_s):
        r_c_s = torch.tensor(r_c_s, dtype=torch.float32)
        p = self.pitch_predicton(r_c_s)[0]

        p_forward = p[:, :, :32]
        p_backward = p[:, :, 32:]

        _p = torch.cat((p_forward[:, 7::8, :], p_backward[:, ::8, :]), dim=-1)

        pitch_p = self.linear(_p)

        return pitch_p

class pitch_predictor(nn.Module):
    def __init__(self):
        super(pitch_predictor, self).__init__()
        self.pitch_predicton = nn.LSTM(input_size=24, hidden_size=32, num_layers=8, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(64, 1)

    def forward(self, r_c_s):
        r_c_s = torch.tensor(r_c_s, dtype=torch.float32)
        p = self.pitch_predicton(r_c_s)[0]

        pitch_p = self.linear(p)

        return pitch_p


if __name__ == '__main__':
    model = pitch_predictor()
    x = torch.rand(10, 192, 24)
    pitch_p = model.forward(x)