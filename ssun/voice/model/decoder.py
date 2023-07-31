import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder).__init__()
        self.lstm_d = nn.LSTM(input_size = 164, hidden_size = 512, num_layers = 3, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(1024, 80, bias=True)

    def set_input(self, x):
        self.input = x

    def forward(self):
        output = self.lstm_d(self.input)[0]
        decoder_output = self.linear(output)

        return decoder_output