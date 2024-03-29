import torch
import torch.nn as nn

class pitch_predictor_original(nn.Module):
    def __init__(self):
        super(pitch_predictor_original, self).__init__()
        self.pitch_bid_LSTM = nn.LSTM(input_size=24, hidden_size=16, num_layers=4, batch_first=True, bidirectional=True)
        # self.pitch_bid_LSTM = nn.LSTM(input_size=18, hidden_size=16, num_layers=4, batch_first=True, bidirectional=True)
        self.pitch_LSTM = nn.LSTM(input_size=32, hidden_size=1, num_layers=4, batch_first=True, bidirectional=False)

        self.pitch_bid_linear = nn.Linear(16, 6)

    def forward(self, r_c_s):
        r_c_s = torch.tensor(r_c_s, dtype=torch.float32).clone().detach()

        p = self.pitch_bid_LSTM(r_c_s)[0]

        p_forward = p[:, :, :32]
        p_backward = p[:, :, 32:]

        _p = torch.cat((p_forward[:, 7::8, :], p_backward[:, ::8, :]), dim=-1)

        pitch_p = self.pitch_LSTM(_p)[0]

        return pitch_p

class pitch_predictor_without_content(nn.Module):
    def __init__(self):
        super(pitch_predictor_without_content, self).__init__()
        # self.pitch_bid_LSTM = nn.LSTM(input_size=22, hidden_size=16, num_layers=4, batch_first=True, bidirectional=True)
        self.pitch_bid_LSTM = nn.LSTM(input_size=16, hidden_size=16, num_layers=4, batch_first=True, bidirectional=True)
        self.pitch_LSTM = nn.LSTM(input_size=32, hidden_size=1, num_layers=4, batch_first=True, bidirectional=False)

        self.pitch_bid_linear = nn.Linear(16, 6)

    def forward(self, r_c_s):
        r_c_s = torch.tensor(r_c_s, dtype=torch.float32)

        p = self.pitch_bid_LSTM(r_c_s)[0]

        p_forward = p[:, :, :32]
        p_backward = p[:, :, 32:]

        _p = torch.cat((p_forward[:, 7::8, :], p_backward[:, ::8, :]), dim=-1)

        pitch_p = self.pitch_LSTM(_p)[0]

        return pitch_p

