import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True):
        super(Conv_layer, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.Conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, params):
        return self.Conv_layer(params)

class Er(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_r = nn.Sequential(Conv_layer(in_channels = 8, out_channels = 128, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.GroupNorm(num_groups = 8, num_channels = 128))
        self.lstm_r = nn.LSTM(input_size = 128, hidden_size = 1, num_layers = 1, batch_first=True, bidirectional=True)

    def forward(self, r, mask):
        for conv_r in self.conv_r:
            r = F.relu(conv_r(r))
        r = r.transpose(1, 2)

        self.lstm_r.flatten_parameters()
        outputs = self.lstm_r(r)[0]

        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, :1]
        out_backward = outputs[:, :, 1:]

        codes_r = torch.cat((out_forward[:, 7::8, :], out_backward[:, ::8, :]), dim=-1)

        return codes_r

class Ec_Ef(nn.Module):
    def __init__(self):
        super().__init__()
        # Ec architecture
        self.conv_c = nn.Sequential(Conv_layer(in_channels = 8, out_channels = 512, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(in_channels = 512, out_channels = 512, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(in_channels = 512, out_channels = 512, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.GroupNorm(num_groups = 32, num_channels = 512))
        self.lstm_c = nn.LSTM(input_size = 512, hidden_size = 8, num_layers = 1, batch_first=True, bidirectional=True)

        # Ef architecture
        self.conv_f = nn.Sequential(Conv_layer(in_channels = 257, out_channels = 256, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(in_channels = 256, out_channels = 256, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(in_channels = 256, out_channels = 256, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.GroupNorm(num_groups = 16, num_channels = 256))
        self.lstm_f = nn.LSTM(input_size = 256, hidden_size = 32,  num_layers = 1, batch_first=True, bidirectional=True)

        self.interp = InterpLnr()

    def forward(self, c_f):
        c = c_f[:, :8, :]
        f = c_f[:, 8:, :]

        for conv_c, conv_f in zip(self.conv_c, self.conv_f):
            c = F.relu(conv_c(c))
            f = F.relu(conv_f(f))

            c_f = torch.cat((c, f), dim=1).transpose(1, 2)
            c_f = self.interp(c_f, self.len_org.expand(c.size(0)))
            c_f = c_f.transpose(1, 2)
            c = c_f[:, :512, :]
            f = c_f[:, 512:, :]
            print('end')

        c_f = c_f.transpose(1, 2)
        c = c_f[:, :, :512]
        f = c_f[:, :, 512:]

        c = self.lstm_c(c)[0]
        f = self.lstm_f(f)[0]

        c_forward = c[:, :, :8]
        c_backward = c[:, :, 8:]

        f_forward = f[:, :, :32]
        f_backward = f[:, :, 32:]

        codes_c = torch.cat((c_forward[:, 7::8, :], c_backward[:, ::8, :]), dim=-1)
        codes_f = torch.cat((f_forward[:, 7::8, :], f_backward[:, ::8, :]), dim=-1)

        return codes_c, codes_f