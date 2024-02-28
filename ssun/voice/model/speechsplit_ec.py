import torch
import torch.nn as nn
import torch.nn.functional as F
from .hparams import hparams

class speechsplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.Er = Er()
        self.Ec = Ec()
        self.Ep = Ep()
        self.D = D()

        self.freq_c = hparams.freq # 8
        self.freq_r = hparams.freq_2 # 8
        self.freq_f = hparams.freq_3 # 8

    def forward(self, x_f0, x_org, c_trg):
        # input : x_f0_intrp_org.shape : torch.Size([2, 192, 337]), x_real_org.shape : torch.Size([2, 192, 80]), emb_org.shape : torch.Size([2, 82])
        x_1 = x_f0.transpose(2, 1)

        c = x_1[:, :80, :]

        codes_c , c_forward, c_backward = self.Ec(c)
        codes_c_1 = torch.cat((c_forward[:, 11::12, :], c_backward[:, ::12, :]), dim=-1)
        _, codes_f = self.Ep(x_1)
        codes_c_r = codes_c.repeat_interleave(self.freq_c, dim=1)
        codes_f_r = codes_f.repeat_interleave(self.freq_f, dim=1)

        x_2 = x_org.transpose(2, 1)
        codes_r = self.Er(x_2, mask=None)
        codes_r_r = codes_r.repeat_interleave(self.freq_r, dim=1)

        encoder_outputs = torch.cat((codes_c_r, codes_r_r, codes_f_r, c_trg.unsqueeze(1).expand(-1, x_1.size(-1), -1)), dim=-1)
        mel_outputs = self.D(encoder_outputs)

        return codes_c_1.repeat_interleave(16, dim=1).transpose(2, 1)

    def rhythm(self, x_org):
        x_2 = x_org.transpose(2, 1)
        codes_2 = self.Er(x_2, mask=None)
        return codes_2


class Er(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_r = hparams.freq_2 # 8
        self.dim_neck_r = hparams.dim_neck_2 # 1
        self.dim_enc_r = hparams.dim_enc_2 # 128

        self.dim_freq = hparams.dim_freq # 80
        self.chs_grp = hparams.chs_grp # 16

        convolutions = []
        for i in range(1):
            conv_layer = nn.Sequential(
                Conv_layer(self.dim_freq if i == 0 else self.dim_enc_r,
                         self.dim_enc_r,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_r // self.chs_grp, self.dim_enc_r))
            convolutions.append(conv_layer)
        self.conv_r = nn.ModuleList(convolutions)
        self.lstm_r = nn.LSTM(self.dim_enc_r, self.dim_neck_r, 1, batch_first=True, bidirectional=True)

    def forward(self, r, mask):
        for conv_r in self.conv_r:
            r = F.relu(conv_r(r))
        r = r.transpose(1, 2)

        self.lstm_r.flatten_parameters()
        outputs = self.lstm_r(r)[0]

        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, :self.dim_neck_r]
        out_backward = outputs[:, :, self.dim_neck_r:]

        codes_r = torch.cat((out_forward[:, self.freq_r-1::self.freq_r, :], out_backward[:, ::self.freq_r, :]), dim=-1)

        return codes_r

class Ec(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_neck = hparams.dim_neck
        self.freq = hparams.freq
        self.freq_3 = hparams.freq_3
        self.dim_enc = hparams.dim_enc
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_freq = hparams.dim_freq
        self.chs_grp = hparams.chs_grp
        self.register_buffer('len_org', torch.tensor(hparams.max_len_pad))
        self.dim_neck_3 = hparams.dim_neck_3
        self.dim_f0 = hparams.dim_f0

        # Ec architecture
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                Conv_layer(self.dim_freq if i == 0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc // self.chs_grp, self.dim_enc))
            convolutions.append(conv_layer)
        self.convolutions_c = nn.ModuleList(convolutions)

        self.lstm_c = nn.LSTM(self.dim_enc, self.dim_neck, 2, batch_first=True, bidirectional=True)
        self.interp = InterpLnr(hparams)

    def forward(self, c):
        for conv_c in self.convolutions_c:
            c = F.relu(conv_c(c))
            c = c.transpose(1, 2)
            c = self.interp(c, self.len_org.expand(c.size(0)))
            c = c.transpose(1, 2)

        c = c.transpose(1, 2)
        c = self.lstm_c(c)[0]

        c_forward = c[:, :, :self.dim_neck]
        c_backward = c[:, :, self.dim_neck:]

        codes_c = torch.cat((c_forward[:, self.freq - 1::self.freq, :], c_backward[:, ::self.freq, :]), dim=-1)  # codes_c.shape : torch.Size([2, 24, 16])

        return codes_c, c_forward, c_backward

class Ep(nn.Module):
    def __init__(self):
        super().__init__()

        self.dim_neck = hparams.dim_neck
        self.freq = hparams.freq
        self.freq_3 = hparams.freq_3
        self.dim_enc = hparams.dim_enc
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_freq = hparams.dim_freq
        self.chs_grp = hparams.chs_grp
        self.register_buffer('len_org', torch.tensor(hparams.max_len_pad))
        self.dim_neck_3 = hparams.dim_neck_3
        self.dim_f0 = hparams.dim_f0

        # convolutions for code 1
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                Conv_layer(self.dim_freq if i == 0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc // self.chs_grp, self.dim_enc))
            convolutions.append(conv_layer)
        self.convolutions_1 = nn.ModuleList(convolutions)

        self.lstm_1 = nn.LSTM(self.dim_enc, self.dim_neck, 2, batch_first=True, bidirectional=True)

        # convolutions for f0
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                Conv_layer(self.dim_f0 if i == 0 else self.dim_enc_3,
                         self.dim_enc_3,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_3 // self.chs_grp, self.dim_enc_3))
            convolutions.append(conv_layer)
        self.convolutions_2 = nn.ModuleList(convolutions)

        self.lstm_2 = nn.LSTM(self.dim_enc_3, self.dim_neck_3, 1, batch_first=True, bidirectional=True)

        self.interp = InterpLnr(hparams)

    def forward(self, x_f0):

        x = x_f0[:, :self.dim_freq, :]
        f0 = x_f0[:, self.dim_freq:, :]

        for conv_1, conv_2 in zip(self.convolutions_1, self.convolutions_2):
            x = F.relu(conv_1(x))
            f0 = F.relu(conv_2(f0))
            x_f0 = torch.cat((x, f0), dim=1).transpose(1, 2)
            x_f0 = self.interp(x_f0, self.len_org.expand(x.size(0)))
            x_f0 = x_f0.transpose(1, 2)
            x = x_f0[:, :self.dim_enc, :]
            f0 = x_f0[:, self.dim_enc:, :]

        x_f0 = x_f0.transpose(1, 2)
        x = x_f0[:, :, :self.dim_enc]
        f0 = x_f0[:, :, self.dim_enc:]

        # code 1
        x = self.lstm_1(x)[0]
        f0 = self.lstm_2(f0)[0]

        x_forward = x[:, :, :self.dim_neck]
        x_backward = x[:, :, self.dim_neck:]

        f0_forward = f0[:, :, :self.dim_neck_3]
        f0_backward = f0[:, :, self.dim_neck_3:]

        codes_x = torch.cat((x_forward[:, self.freq - 1::self.freq, :], x_backward[:, ::self.freq, :]), dim=-1)

        codes_f0 = torch.cat((f0_forward[:, self.freq_3 - 1::self.freq_3, :],
                              f0_backward[:, ::self.freq_3, :]), dim=-1)

        return codes_x, codes_f0
class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_neck = hparams.dim_neck # 8
        self.dim_neck_2 = hparams.dim_neck_2 # 1
        self.dim_emb = hparams.dim_spk_emb # 83
        self.dim_freq = hparams.dim_freq # 8
        self.dim_neck_3 = hparams.dim_neck_3 # 32

        self.lstm_d = nn.LSTM(104, 512, 3, batch_first=True, bidirectional=True) #####
        self.linear = nn.Linear(1024, self.dim_freq, bias=True)

    def forward(self, x):
        output = self.lstm_d(x.float())[0]
        decoder_output = self.linear(output)

        return decoder_output

class Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(Conv_layer, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class InterpLnr(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.max_len_seq = hparams.max_len_seq # 128
        self.max_len_pad = hparams.max_len_pad # 192

        self.min_len_seg = hparams.min_len_seg # 19
        self.max_len_seg = hparams.max_len_seg # 32

        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def pad_sequences(self, sequences):
        channel_dim = sequences[0].size()[-1] # channel_dim : 81
        out_dims = (len(sequences), self.max_len_pad, channel_dim) # out_dims : (2, 192, 81)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0) # out_tensor.shape : torch.Size([2, 192, 81])

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[:self.max_len_pad]

        return out_tensor

    def forward(self, x, len_seq):
        if not self.training:
            return x
        device = x.device
        batch_size = x.size(0)

        # a=torch.arange(self.max_len_seg * 2, device=device) # a.shape : torch.Size([64])
        # b=a.unsqueeze(0) # b.shape : torch.Size([1, 64])
        # c=b.expand(batch_size * self.max_num_seg, -1) # c.shape : torch.Size([14, 64])
        indices = torch.arange(self.max_len_seg * 2, device=device).unsqueeze(0).expand(batch_size * self.max_num_seg, -1)

        # e=torch.rand(batch_size * self.max_num_seg, device=device) # e.shape : torch.Size([14])
        scales = torch.rand(batch_size * self.max_num_seg, device=device) + 0.5 # scales.shape : torch.Size([14])

        idx_scaled = indices / scales.unsqueeze(-1) # scales.unsqueeze(-1).shape : torch.Size([14, 1]), idx_scaled.shape : torch.Size([14, 64])
        idx_scaled_fl = torch.floor(idx_scaled) # idx_scaled_fl.shape : torch.Size([14, 64])
        lambda_ = idx_scaled - idx_scaled_fl # lambda_.shape : torch.Size([14, 64])

        len_seg = torch.randint(low=self.min_len_seg, high=self.max_len_seg, size=(batch_size * self.max_num_seg, 1), device=device)
        # len_seg.shape : torch.Size([14, 1])

        idx_mask = idx_scaled_fl < (len_seg - 1) # idx_mask.shape : torch.Size([14, 64])

        offset = len_seg.view(batch_size, -1).cumsum(dim=-1) # len_seg.view(batch_size, -1).shape : torch.Size([2, 7]), offset.shape : torch.Size([2, 7])

        offset = F.pad(offset[:, :-1], (1, 0), value=0).view(-1, 1) # offset.shape : torch.Size([14, 1])

        idx_scaled_org = idx_scaled_fl + offset # idx_scaled_org.shape : torch.Size([14, 64])

        len_seq_rp = torch.repeat_interleave(len_seq, self.max_num_seg) # len_seq_rp.shape : torch.Size([14])
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1) # idx_mask_org.shape : torch.Size([14, 64])

        idx_mask_final = idx_mask & idx_mask_org # idx_mask_final.shape : torch.Size([14, 64])

        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)
        # idx_mask_final.sum(dim=-1).shape : torch.Size([14]),
        # idx_mask_final.sum(dim=-1).view(batch_size, -1).shape : torch.Size([2, 7]),
        # counts.shape : torch.Size([2])

        index_1 = torch.repeat_interleave(torch.arange(batch_size, device=device), counts) # index_1.shape : torch.Size([counts[0]+counts[1]])

        index_2_fl = idx_scaled_org[idx_mask_final].long() # index_2_fl.shape : torch.Size([counts[0]+counts[1]])
        index_2_cl = index_2_fl + 1 # index_2_cl.shape : torch.Size([counts[0]+counts[1]])

        y_fl = x[index_1, index_2_fl, :] # y_fl.shape : torch.Size([counts[0]+counts[1], 81])
        y_cl = x[index_1, index_2_cl, :] # y_cl.shape : torch.Size([counts[0]+counts[1], 81])
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)
        # lambda_[idx_mask_final].shape : torch.Size([counts[0]+counts[1]]), lambda_f.shape : torch.Size([counts[0]+counts[1], 1])

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl # y.shape : torch.Size([counts[0]+counts[1], 81])

        sequences = torch.split(y, counts.tolist(), dim=0)
        # type(sequences) : tuple, len(sequences) : 2, sequences[0].shape : torch.Size([counts[0], 81]), sequences[1].shape : torch.Size([counts[1], 81])
        seq_padded = self.pad_sequences(sequences) # seq_padded.shape : torch.Size([2, 192, 81])

        return seq_padded


class InterpLnr_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_len_seq = hparams.max_len_seq # 128
        self.max_len_pad = hparams.max_len_pad # 192

        self.min_len_seg = hparams.min_len_seg # 19
        self.max_len_seg = hparams.max_len_seg # 32

        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def pad_sequences(self, sequences):
        channel_dim = sequences[0].size()[-1] # channel_dim : 81
        out_dims = (len(sequences), self.max_len_pad, channel_dim) # out_dims : (2, 192, 81)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0) # out_tensor.shape : torch.Size([2, 192, 81])

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[:self.max_len_pad]

        return out_tensor

    def forward(self, x, len_seq):
        if not self.training:
            return x
        device = x.device
        batch_size = x.size(0)

        # a=torch.arange(self.max_len_seg * 2, device=device) # a.shape : torch.Size([64])
        # b=a.unsqueeze(0) # b.shape : torch.Size([1, 64])
        # c=b.expand(batch_size * self.max_num_seg, -1) # c.shape : torch.Size([14, 64])
        indices = torch.arange(self.max_len_seg * 2, device=device).unsqueeze(0).expand(batch_size * self.max_num_seg, -1)

        # e=torch.rand(batch_size * self.max_num_seg, device=device) # e.shape : torch.Size([14])
        scales = torch.rand(batch_size * self.max_num_seg, device=device) + 0.5 # scales.shape : torch.Size([14])

        idx_scaled = indices / scales.unsqueeze(-1) # scales.unsqueeze(-1).shape : torch.Size([14, 1]), idx_scaled.shape : torch.Size([14, 64])
        idx_scaled_fl = torch.floor(idx_scaled) # idx_scaled_fl.shape : torch.Size([14, 64])
        lambda_ = idx_scaled - idx_scaled_fl # lambda_.shape : torch.Size([14, 64])

        len_seg = torch.randint(low=self.min_len_seg, high=self.max_len_seg, size=(batch_size * self.max_num_seg, 1), device=device)
        # len_seg.shape : torch.Size([14, 1])

        idx_mask = idx_scaled_fl < (len_seg - 1) # idx_mask.shape : torch.Size([14, 64])

        offset = len_seg.view(batch_size, -1).cumsum(dim=-1) # len_seg.view(batch_size, -1).shape : torch.Size([2, 7]), offset.shape : torch.Size([2, 7])

        offset = F.pad(offset[:, :-1], (1, 0), value=0).view(-1, 1) # offset.shape : torch.Size([14, 1])

        idx_scaled_org = idx_scaled_fl + offset # idx_scaled_org.shape : torch.Size([14, 64])

        len_seq_rp = torch.repeat_interleave(len_seq, self.max_num_seg) # len_seq_rp.shape : torch.Size([14])
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1) # idx_mask_org.shape : torch.Size([14, 64])

        idx_mask_final = idx_mask & idx_mask_org # idx_mask_final.shape : torch.Size([14, 64])

        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)
        # idx_mask_final.sum(dim=-1).shape : torch.Size([14]),
        # idx_mask_final.sum(dim=-1).view(batch_size, -1).shape : torch.Size([2, 7]),
        # counts.shape : torch.Size([2])

        index_1 = torch.repeat_interleave(torch.arange(batch_size, device=device), counts) # index_1.shape : torch.Size([counts[0]+counts[1]])

        index_2_fl = idx_scaled_org[idx_mask_final].long() # index_2_fl.shape : torch.Size([counts[0]+counts[1]])
        index_2_cl = index_2_fl + 1 # index_2_cl.shape : torch.Size([counts[0]+counts[1]])

        y_fl = x[index_1, index_2_fl, :] # y_fl.shape : torch.Size([counts[0]+counts[1], 81])
        y_cl = x[index_1, index_2_cl, :] # y_cl.shape : torch.Size([counts[0]+counts[1], 81])
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)
        # lambda_[idx_mask_final].shape : torch.Size([counts[0]+counts[1]]), lambda_f.shape : torch.Size([counts[0]+counts[1], 1])

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl # y.shape : torch.Size([counts[0]+counts[1], 81])

        sequences = torch.split(y, counts.tolist(), dim=0)
        # type(sequences) : tuple, len(sequences) : 2, sequences[0].shape : torch.Size([counts[0], 81]), sequences[1].shape : torch.Size([counts[1], 81])
        seq_padded = self.pad_sequences(sequences) # seq_padded.shape : torch.Size([2, 192, 81])

        return seq_padded