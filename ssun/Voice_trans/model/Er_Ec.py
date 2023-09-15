import sys, os
sys.path.append("..")
import torch
import torch.nn as nn


class Er_Ec(nn.Module):
    def __init__(self, opt):
        super(Er_Ec,self).__init__()

        if not opt.debugging:
            from ssun.Voice_trans.model.encoder import Er, Ec
        else:
            from Voice_trans.model.encoder import Er, Ec

        self.Er = Er()
        self.Ec = Ec()

    def forward(self, voice, sp_id):
        rhythm = self.Er(voice.transpose(2,1))
        content = self.Ec(voice.transpose(2,1))

        rhythm = rhythm.repeat_interleave(8, dim=1)
        content = content.repeat_interleave(8, dim=1)

        r_c_s = torch.cat((rhythm, content, sp_id.unsqueeze(1).expand(-1, voice.transpose(2, 1).size(-1), -1)), dim=-1)

        return rhythm, content, r_c_s
