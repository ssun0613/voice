import torch
import torch.nn as nn
import numpy as np

from decoder_f_sub import FFTBlock
from text.symbols import symbols

def get_mask_from_lengths(lengths, max_len=None): # src_lens --> lengths, max_src_len --> max_len
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

class Decoder(nn.Module):
    """ Decoder """
    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = 1001 # config["max_seq_len"] + 1
        d_word_vec = 256 # config["transformer"]["decoder_hidden"]
        n_layers = 6 # config["transformer"]["decoder_layer"]
        n_head = 2 # config["transformer"]["decoder_head"]
        d_k = d_v = (d_word_vec//n_head) # (config["transformer"]["decoder_hidden"] // config["transformer"]["decoder_head"])
        d_model = 256 # config["transformer"]["decoder_hidden"]
        d_inner = 1024 # config["transformer"]["conv_filter_size"]
        kernel_size = [9, 1] # config["transformer"]["conv_kernel_size"]
        dropout = 0.2 # config["transformer"]["decoder_dropout"]

        self.max_seq_len = 1000 # config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0), requires_grad=False)

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout) for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):
        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model)[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(enc_seq.device)
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(dec_output, mask=mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask

if __name__=="__main__":
    mel_lens = torch.tensor([611], device='cuda:0')
    max_mel_len = 611

    mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None)

    output = torch.rand([1, 611, 256], device='cuda:0', grad_fn=<StackBackward0>)

    output, mel_masks = Decoder(output, mel_masks )



