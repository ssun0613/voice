import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder_s(nn.Module):
    def __init__(self):
        super(Decoder_s, self).__init__()
        self.lstm_d = nn.LSTM(input_size = 274, hidden_size = 512, num_layers = 3, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(1024, 80, bias=True)


    def forward(self, x):
        output = self.lstm_d(x)[0]
        decoder_output = self.linear(output)

        return decoder_output