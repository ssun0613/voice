import sys, os
sys.path.append("..")
import torch
import torch.nn as nn


class generator(nn.Module):
    def __init__(self, opt, device):
        super(generator,self).__init__()
        if not opt.debugging:
            from ssun.Voice_trans.model.encoder import Er, Ec
            from ssun.Voice_trans.model.decoder_s import Decoder_s as Ds
            # from ssun.Voice_trans.model.decoder_f import Decoder_f as Df
            from ssun.Voice_trans.model.pitch_predictor import pitch_predictor as P
        else:
            from Voice_trans.model.encoder import Er, Ec
            from Voice_trans.model.decoder_s import Decoder_s as Ds
            # from Voice_trans.model.decoder_f import Decoder_f as Df
            from Voice_trans.model.pitch_predictor import pitch_predictor as P

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

        rhythm_r = self.Er(mel_output.transpose(2, 1)) # used to calculate rhythm reconstruction loss
        content_r = self.Ec(mel_output.transpose(2, 1)) # used to calculate content reconstruction loss

        rhythm_r_repeat = rhythm_r.repeat_interleave(8, dim=1)
        content_r_repeat = content_r.repeat_interleave(8, dim=1)

        r_c_p_r = torch.cat((rhythm_r_repeat, content_r_repeat, sp_id.unsqueeze(1).expand(-1, voice.transpose(2,1).size(-1), -1)), dim=-1) # used to calculate pitch reconstruction loss

        pitch_p_r, pitch_embedding_r = self.P(r_c_p_r)

        # return mel_output, pitch_p, rhythm, content, rhythm_r, content_r
        return mel_output, pitch_p, pitch_embedding, rhythm, content, rhythm_r, content_r, pitch_embedding_r
