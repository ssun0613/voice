import sys, os
sys.path.append("..")
import torch
import torch.nn as nn


class voice_trans(nn.Module):
    def __init__(self, opt, device):
        super(voice_trans,self).__init__()
        from ssun.Voice_trans.model.encoder import Er, Ec
        from ssun.Voice_trans.model.decoder_s import Decoder_s as Ds
        # from ssun.Voice_trans.model.decoder_f import Decoder_f as Df
        from ssun.Voice_trans.model.pitch_predictor import pitch_predictor as P

        self.Er = Er()
        self.Ec = Ec()
        self.Ds = Ds()
        # self.Df = Df()
        self.P = P()

    def forward(self, voice, sp_id):
        rhythm = self.Er(voice.transpose(2,1))
        content = self.Ec(voice.transpose(2,1))

        rhythm_repeat = rhythm.repeat_interleave(8, dim=1)
        content_repeat = content.repeat_interleave(8, dim=1)

        r_c_s = torch.cat((rhythm_repeat, content_repeat, sp_id.unsqueeze(1).expand(-1, voice.transpose(2,1).size(-1), -1)), dim=-1)
        pitch_p, pitch_embedding = self.P(r_c_s)

        r_c_p = torch.cat((rhythm_repeat, content_repeat, pitch_embedding), dim=-1)
        mel_output = self.Ds(r_c_p)

        rhythm_l = self.Er(mel_output.transpose(2, 1)) # used to calculate rhythm reconstruction loss
        content_l = self.Ec(mel_output.transpose(2, 1)) # used to calculate content reconstruction loss

        return mel_output, pitch_p, rhythm, content, rhythm_l, content_l
