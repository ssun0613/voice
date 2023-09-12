import sys, os
sys.path.append("..")
import torch
import torch.nn as nn

from ssun.Voice_trans.model.encoder import Er, Ec
from ssun.Voice_trans.model.decoder_s import Decoder_s as Ds
# from ssun.Voice_trans.model.decoder_f import Decoder_f as Df
from ssun.Voice_trans.model.pitch_predictor import pitch_predictor as P

class voice_trans(nn.Module):
    def __init__(self):
        super(voice_trans,self).__init__()
        self.Er = Er()
        self.Ec = Ec()

        self.Ds = Ds()
        # self.Df = Df()

        self.P = P()

    def forward(self, voice, sp_id):
        rhythm = self.Er(voice.transpose(2,1))
        content = self.Ec(voice.transpose(2,1))

        rhythm = rhythm.repeat_interleave(8, dim=1)
        content = content.repeat_interleave(8, dim=1)

        r_c_s = torch.cat((rhythm, content,sp_id.unsqueeze(1).expand(-1, voice.transpose(2,1).size(-1), -1)), dim=-1)

        pitch_p, pitch_embedding = self.P(r_c_s.transpose(2,1))

        r_c_p = torch.cat((rhythm, content, pitch_embedding), dim=-1)

        mel_output = self.Ds(r_c_p)

        return mel_output
