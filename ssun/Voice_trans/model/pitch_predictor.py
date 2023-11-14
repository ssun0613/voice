import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class pitch_predictor(nn.Module):
    def __init__(self):
        super(pitch_predictor, self).__init__()
        self.pitch_bid_LSTM = nn.LSTM(input_size=24, hidden_size=16, num_layers=4, batch_first=True, bidirectional=True)
        self.pitch_LSTM = nn.LSTM(input_size=32, hidden_size=1, num_layers=4, batch_first=True, bidirectional=False)

    def forward(self, r_c_s):
        r_c_s = torch.tensor(r_c_s, dtype=torch.float32)

        p = self.pitch_bid_LSTM(r_c_s)[0]
        pitch_p = self.pitch_LSTM(p)[0]

        return pitch_p


if __name__ == '__main__':
    model = pitch_predictor()
    x = torch.rand(10, 192, 24)
    pitch_p = model.forward(x)
