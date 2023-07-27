import os
import numpy as np
import librosa
import librosa.display
import glob

import torch
from torch.utils import data
from torch.utils.data.sampler import Sampler

LABEL = {'africa': 0, 'australia': 1, 'canada' : 2, 'england' : 3, 'hongkong' : 4, 'us' : 5}

class accent():
    def __init__(self, dataset_path, is_training=True, mode='speechsplit'):
        self.dataset_dir = dataset_path
        self.is_training = is_training
        self.mode = mode
        self.data_path = self.data_load()
        self.dataset_size = len(self.data_path)
        # self.source_path = source_path
        # self.target_path = source_path

    def data_load(self):
        data = sorted(glob.glob(self.dataset_dir + "train/*/*.wav"))

        return data

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # if self.mode =='speechsplit':
        #     data_source, data_target = self.source_target(index)
        #     assert data_sr != None, "sr is None, cheak"
        #     label = self.data_label(self.data_path[index % self.dataset_size])
        #     return data_source, data_target, lable

        # else:
            data_wav, data_sr = librosa.load(self.data_path[index % self.dataset_size], sr=16000)
            assert data_sr!=None, "sr is None, cheak"
            label = self.data_label(self.data_path[index % self.dataset_size])

            data_voice_mel = librosa.feature.melspectrogram(y=data_wav, sr=data_sr) # mel-spectrogram
            data_voice_mel_s = librosa.power_to_db(data_voice_mel, ref=np.max)
            data_voice = librosa.feature.mfcc(S=data_voice_mel_s, sr=data_sr, n_mfcc=20, n_fft=400, hop_length=260) # mfcc

            return data_voice, label

    def source_target(self, index):
        data_wav, data_sr = librosa.load(self.data_path[index % self.dataset_size], sr=16000)
        data_source, data_target = librosa.feature.mfcc(data_wav)
        return data_source, data_target

    def data_label(self, label_path):
        data_label = label_path.split('/')[5]
        data_label_lower = data_label.lower()
        label = np.zeros(len(LABEL))
        label[LABEL[data_label_lower]] = 1.0
        return label

