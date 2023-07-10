import torch
import torch.nn as nn
import torch.nn.functional as F

class speechsplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.Er = Er()
        self.Ec_Ef = Ec_Ef()
        self.D = D()

        self.freq_c = 8 # hparams.freq
        self.freq_r = 8 # hparams.freq_2
        self.freq_f = 8 # hparams.freq_3

    def forward(self, x_f0, x_org, c_trg):
        # input : x_f0_intrp_org.shape : torch.Size([2, 192, 337]), x_real_org.shape : torch.Size([2, 192, 80]), emb_org.shape : torch.Size([2, 82])
        x_1 = x_f0.transpose(2, 1)
        codes_c, codes_f = self.Ec_Ef(x_1)
        codes_c_r = codes_c.repeat_interleave(self.freq_c, dim=1)
        codes_f_r = codes_f.repeat_interleave(self.freq_f, dim=1)

        x_2 = x_org.transpose(2, 1)
        codes_r = self.Er(x_2, mask=None)
        codes_r_r = codes_r.repeat_interleave(self.freq_r, dim=1)

        encoder_outputs = torch.cat((codes_c_r, codes_r_r, codes_f_r, c_trg.unsqueeze(1).expand(-1, x_1.size(-1), -1)), dim=-1)
        mel_outputs = self.D(encoder_outputs)

        return mel_outputs

    def rhythm(self, x_org):
        x_2 = x_org.transpose(2, 1)
        codes_2 = self.Er(x_2, mask=None)
        return codes_2

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
        self.freq_r = 8 # hparams.freq_2
        self.dim_neck_r = 1 # hparams.dim_neck_2
        self.dim_enc_r = 128 # hparams.dim_enc_2

        self.dim_freq = 8 # hparams.dim_freq
        self.chs_grp = 16 # hparams.chs_grp

        self.conv_r = nn.Sequential(Conv_layer(self.dim_freq, self.dim_enc_r, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.GroupNorm(self.dim_enc_r // self.chs_grp, self.dim_enc_r))
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

class Ec_Ef(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_freq = 8 # hparams.dim_freq
        self.dim_f0 = 257 # hparams.dim_f0
        self.chs_grp = 16 # hparams.chs_grp
        self.register_buffer('len_org', torch.tensor(192)) # torch.tensor(hparams.max_len_pad

        # hparams that Ec use
        self.freq_c = 8 # hparams.freq
        self.dim_neck_c = 8 # hparams.dim_neck
        self.dim_enc_c = 512 # hparams.dim_enc

        # hparams that Ef use
        self.freq_f = 8 # hparams.freq_3
        self.dim_neck_f = 32 # hparams.dim_neck_3
        self.dim_enc_f = 256 # hparams.dim_enc_3

        # Ec architecture
        self.conv_c = nn.Sequential(Conv_layer(self.dim_freq, self.dim_enc_c, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(self.dim_enc_c, self.dim_enc_c, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(self.dim_enc_c, self.dim_enc_c, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.GroupNorm(self.dim_enc_c//self.chs_grp, self.dim_enc_c))
        self.lstm_c = nn.LSTM(self.dim_enc_c, self.dim_neck_c, 1, batch_first=True, bidirectional=True)

        # Ef architecture
        self.conv_f = nn.Sequential(Conv_layer(self.dim_f0, self.dim_enc_f, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(self.dim_enc_f, self.dim_enc_f, kernel_size=5, stride=1, padding=2, dilation=1),
                                    Conv_layer(self.dim_enc_f, self.dim_enc_f, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.GroupNorm(self.dim_enc_f//self.chs_grp, self.dim_enc_f))
        self.lstm_f = nn.LSTM(self.dim_enc_f, self.dim_neck_f, 1, batch_first=True, bidirectional=True)

        self.interp = InterpLnr()

    def forward(self, c_f):
        c = c_f[:, :self.dim_freq, :]
        f = c_f[:, self.dim_freq:, :]

        for conv_c, conv_f in zip(self.conv_c, self.conv_f):
            c = F.relu(conv_c(c))
            f = F.relu(conv_f(f))

            c_f = torch.cat((c, f), dim=1).transpose(1, 2)
            c_f = self.interp(c_f, self.len_org.expand(c.size(0)))
            c_f = c_f.transpose(1, 2)
            c = c_f[:, :self.dim_enc_c, :]
            f = c_f[:, self.dim_enc_c:, :]
            print('end')

        c_f = c_f.transpose(1, 2)
        c = c_f[:, :, :self.dim_enc_c]
        f = c_f[:, :, self.dim_enc_c:]

        c = self.lstm_c(c)[0]
        f = self.lstm_f(f)[0]

        c_forward = c[:, :, :self.dim_neck_c]
        c_backward = c[:, :, self.dim_neck_c:]

        f_forward = f[:, :, :self.dim_neck_f]
        f_backward = f[:, :, self.dim_neck_f:]

        codes_c = torch.cat((c_forward[:, self.freq_c-1::self.freq_c, :], c_backward[:, ::self.freq_c, :]), dim=-1)
        codes_f = torch.cat((f_forward[:, self.freq_f-1::self.freq_f, :], f_backward[:, ::self.freq_f, :]), dim=-1)

        return codes_c, codes_f

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_neck = 8 # hparams.dim_neck
        self.dim_neck_2 = 1 # hparams.dim_neck_2
        self.dim_emb = 82 # hparams.dim_spk_emb
        self.dim_freq = 80 # hparams.dim_freq
        self.dim_neck_3 = 32 # hparams.dim_neck_3

        self.lstm_d = nn.LSTM(self.dim_neck * 2 + self.dim_neck_2 * 2 + self.dim_neck_3 * 2 + self.dim_emb, 512, 3, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(1024, self.dim_freq, bias=True)

    def forward(self, x):
        output = self.lstm_d(x)[0]
        decoder_output = self.linear(output)

        return decoder_output

class InterpLnr(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_len_seq = 128 # hparams.max_len_seq
        self.max_len_pad = 192 # hparams.max_len_pad

        self.min_len_seg = 19 # hparams.min_len_seg
        self.max_len_seg = 32 # hparams.max_len_seg

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