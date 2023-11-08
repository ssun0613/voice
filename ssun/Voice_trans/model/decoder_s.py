import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder_s(nn.Module):
    def __init__(self):
        super(Decoder_s, self).__init__()
        self.lstm_d = nn.LSTM(input_size=82, hidden_size=512, num_layers=3, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(1024, 80, bias=True)


    def forward(self, x):
        # self.lstm_d.state_dict().keys()
        # odict_keys(['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0', 'weight_ih_l0_reverse', 'weight_hh_l0_reverse', 'bias_ih_l0_reverse', 'bias_hh_l0_reverse', 'weight_ih_l1', 'weight_hh_l1', 'bias_ih_l1', 'bias_hh_l1', 'weight_ih_l1_reverse', 'weight_hh_l1_reverse', 'bias_ih_l1_reverse', 'bias_hh_l1_reverse', 'weight_ih_l2', 'weight_hh_l2', 'bias_ih_l2', 'bias_hh_l2', 'weight_ih_l2_reverse', 'weight_hh_l2_reverse', 'bias_ih_l2_reverse', 'bias_hh_l2_reverse'])
        output = self.lstm_d(x)[0]
        decoder_output = self.linear(output)

        return decoder_output