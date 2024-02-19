import torch
import torch.nn as nn


def compute_G_loss(opt, voice, pitch_t, mel_output, pitch_p, rhythm, content, rhythm_r, content_r, pitch_p_r, d_r_mel_out = None):
    loss_m = nn.MSELoss(reduction='sum')
    loss_l = nn.L1Loss(reduction='sum')

    voice_loss = loss_m(voice, mel_output)
    rhythm_loss = loss_l(rhythm, rhythm_r)
    content_loss = loss_l(content, content_r)

    recon_voice_loss = voice_loss + (opt.lambda_r * rhythm_loss) + (opt.lambda_c * content_loss)

    pitch_predition_loss = loss_m(pitch_t, pitch_p)
    pitch_reconstruction_loss = loss_l(pitch_p, pitch_p_r)

    recon_pitch_loss = pitch_predition_loss + (opt.lambda_p * pitch_reconstruction_loss)

    # loss_dr_for_g = (1 - torch.mean((0 - d_r_mel_out) ** 2)) * 100
    loss_dr_for_g = 0

    total_loss_g = recon_voice_loss + recon_pitch_loss + loss_dr_for_g

    return recon_voice_loss, recon_pitch_loss, total_loss_g
def compute_D_loss(d_r_mel_in, d_r_mel_out):
    loss_d_r_r = torch.mean((1 - d_r_mel_in) ** 2)
    loss_d_r_f = torch.mean((0 - d_r_mel_out) ** 2)
    loss_dr = (loss_d_r_r + loss_d_r_f) / 2.0

    return loss_dr