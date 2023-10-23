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
        super(Er, self).__init__()
        # self.conv_r = nn.Sequential(Conv_layer(in_channels = 80, out_channels = 128, kernel_size=5, stride=1, padding=2, dilation=1),
        #                             nn.GroupNorm(num_groups = 8, num_channels = 128))
        self.conv_r = nn.Sequential(Conv_layer(in_channels = 80, out_channels = 128, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(in_channels = 128, out_channels = 128, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(in_channels = 128, out_channels = 128, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(in_channels = 128, out_channels = 128, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.GroupNorm(num_groups = 8, num_channels = 128))
        self.lstm_r = nn.LSTM(input_size = 128, hidden_size = 1, num_layers = 1, batch_first = True, bidirectional = True)
        self.interp = InterpLnr()

    def forward(self, r):
        for conv_r in self.conv_r:
            r = F.relu(conv_r(r))
        r = r.transpose(1, 2)

        self.lstm_r.flatten_parameters()
        outputs = self.lstm_r(r)[0]

        out_forward = outputs[:, :, :1]
        out_backward = outputs[:, :, 1:]

        codes_r = torch.cat((out_forward[:, 7::8, :], out_backward[:, ::8, :]), dim=-1)

        return codes_r

class Ec(nn.Module):
    def __init__(self):
        super(Ec, self).__init__()
        self.register_buffer('len_org', torch.tensor(192))
        # Ec architecture
        self.conv_c = nn.Sequential(Conv_layer(in_channels = 80, out_channels = 512, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(in_channels = 512, out_channels = 512, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(in_channels = 512, out_channels = 512, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(in_channels = 512, out_channels = 512, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.GroupNorm(num_groups = 32, num_channels = 512))
        self.lstm_c = nn.LSTM(input_size = 512, hidden_size = 8, num_layers = 1, batch_first = True, bidirectional = True)
        self.interp = InterpLnr()

    def forward(self, c):
        for conv_c in self.conv_c:
            c = F.relu(conv_c(c))
            c = c.transpose(1, 2)
            c = self.interp(c, self.len_org.expand(c.size(0)))
            c = c.transpose(1, 2)

        c = c.transpose(1, 2)
        c = self.lstm_c(c)[0]

        c_forward = c[:, :, :8]
        c_backward = c[:, :, 8:]

        codes_c = torch.cat((c_forward[:, 7::8, :], c_backward[:, ::8, :]), dim=-1) # codes_c.shape : torch.Size([2, 24, 16])

        return codes_c

class InterpLnr(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_len_seq = 128
        self.max_len_pad = 192

        self.min_len_seg = 19
        self.max_len_seg = 32

        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def pad_sequences(self, sequences):
        channel_dim = sequences[0].size()[-1]
        out_dims = (len(sequences), self.max_len_pad, channel_dim)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[:self.max_len_pad]

        return out_tensor

    def forward(self, x, len_seq):
        if not self.training:
            return x
        device = x.device
        batch_size = x.size(0)

        indices = torch.arange(self.max_len_seg * 2, device=device).unsqueeze(0).expand(batch_size * self.max_num_seg, -1)

        scales = torch.rand(batch_size * self.max_num_seg, device=device) + 0.5

        idx_scaled = indices / scales.unsqueeze(-1)
        idx_scaled_fl = torch.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl

        len_seg = torch.randint(low=self.min_len_seg, high=self.max_len_seg, size=(batch_size * self.max_num_seg, 1), device=device)

        idx_mask = idx_scaled_fl < (len_seg - 1)

        offset = len_seg.view(batch_size, -1).cumsum(dim=-1)
        offset = F.pad(offset[:, :-1], (1, 0), value=0).view(-1, 1)

        idx_scaled_org = idx_scaled_fl + offset

        len_seq_rp = torch.repeat_interleave(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1)

        idx_mask_final = idx_mask & idx_mask_org

        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)

        index_1 = torch.repeat_interleave(torch.arange(batch_size, device=device), counts)
        index_2_fl = idx_scaled_org[idx_mask_final].long()
        index_2_cl = index_2_fl + 1

        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl
        sequences = torch.split(y, counts.tolist(), dim=0)
        seq_padded = self.pad_sequences(sequences) # seq_padded.shape : torch.Size([2, 192, 81])

        return seq_padded

if __name__ == '__main__':
    model = Er()
    x = torch.rand(2, 192, 80).transpose(2, 1)
    model.forward(x)