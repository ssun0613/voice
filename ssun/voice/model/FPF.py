import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from .FPF_modules import TextEncoder, Decoder, VarianceAdaptor, Generator
# from ssun.voice.utils.tools import get_mask_from_lengths
from utils.tools import get_mask_from_lengths

class FastPitchFormant(nn.Module):
    """ FastPitchFormant """
    def __init__(self, preprocess_config, model_config):
        super(FastPitchFormant, self).__init__()
        self.model_config = model_config

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.formant_generator = Generator(model_config)
        self.excitation_generator = Generator(model_config, query_projection=True)
        self.decoder = Decoder(preprocess_config, model_config)

        self.speaker_emb = nn.Embedding(6, 256)

    def forward(self, speakers, pre_c, max_src_len, p_targets=None, mels=None, mel_lens=None, max_mel_len=None, d_targets=None, p_control=1.0, d_control=1.0):
        src_lens = None
        src_masks = (get_mask_from_lengths(src_lens, max_src_len) if src_lens is not None else None)
        mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None)

        speaker_embedding = None
        if self.speaker_emb is not None:
            # speaker_embedding = self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)
            speaker_embedding = self.speaker_emb(speakers).unsqueeze(1).expand(-1, 16, -1)
            pre_c = pre_c + speaker_embedding

        (h, p, p_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks) = self.variance_adaptor(pre_c, speaker_embedding, src_masks, mel_masks, max_mel_len, p_targets, d_targets, p_control, d_control)

        formant_hidden = self.formant_generator(h, mel_masks)
        excitation_hidden = self.excitation_generator(p, mel_masks, hidden_query=h)

        mel_iters, mel_masks = self.decoder(formant_hidden, excitation_hidden, mel_masks)

        return (mel_iters, p_predictions, log_d_predictions, d_rounded, src_masks, mel_masks, src_lens, mel_lens)