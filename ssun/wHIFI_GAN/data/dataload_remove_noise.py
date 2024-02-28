import os
import numpy as np
import glob
import torch
from torch.utils import data
from torch.utils.data.sampler import Sampler

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# LABEL = {'africa': 0, 'australia': 1, 'canada' : 2, 'england' : 3, 'hongkong' : 4, 'us' : 5}
LABEL = {'africa': 1, 'australia': 2, 'canada' : 3, 'england' : 4, 'hongkong' : 5, 'us' : 0}
class accent():
    def __init__(self, dataset_path, is_training=True):
        self.dataset_dir = dataset_path
        self.is_training = is_training
        self.dataset_mel, self.dataset_pitch = self.data_load_npy()
        self.dataset_size = len(self.dataset_mel)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        mel_tmp = np.load(self.dataset_mel[index % self.dataset_size])
        pitch_tmp = np.load(self.dataset_pitch[index % self.dataset_size])
        sp_id = self.data_sp_id(self.dataset_mel[index % self.dataset_size])

        return mel_tmp, pitch_tmp, sp_id

    def data_sp_id(self, label_path):
        data_sp_id = label_path.split('/')[-1].split('_')[0]
        data_sp_id_lower = data_sp_id.lower()
        sp_id = np.zeros(len(LABEL))
        sp_id[LABEL[data_sp_id_lower]] = 1.0

        return sp_id

    def data_load_npy(self):
        mel_data = []
        pitch_data = []

        dataset_mel = sorted(glob.glob(self.dataset_dir + 'mel/*.npy'))
        dataset_pitch = sorted(glob.glob(self.dataset_dir + 'pitch/*.npy'))

        mel_data.append(dataset_mel[2])
        mel_data.append(dataset_mel[-2])

        pitch_data.append(dataset_pitch[2])
        pitch_data.append(dataset_pitch[-2])

        return mel_data, pitch_data

class MyCollator(object):
    def __init__(self):
        self.min_len_seq = 64
        self.max_len_seq = 128
        self.max_len_pad = 192

    def __call__(self, batch):
        new_batch = []
        for token in batch:
            mel, pitch, sp_id = token
            len_crop = np.random.randint(self.min_len_seq, self.max_len_seq + 1, size=2)
            left = np.random.randint(0, len(mel) - len_crop[0], size=2)

            mel_crop = mel[left[0]:left[0] + len_crop[0], :]
            pitch_crop = pitch[left[0]:left[0] + len_crop[0]]

            mel_clip = np.clip(mel_crop, 0, 1)

            mel_pad = np.pad(mel_clip, ((0, self.max_len_pad - mel_clip.shape[0]), (0, 0)), 'constant')
            pitch_pad = np.pad(pitch_crop[:, np.newaxis], ((0, self.max_len_pad - pitch_crop.shape[0]), (0, 0)), 'constant')

            new_batch.append((mel_pad, pitch_pad, len_crop[0], sp_id))

        batch = new_batch

        a, b, c, d= zip(*batch)

        melsp = torch.from_numpy(np.stack(a, axis=0))
        pitch = torch.from_numpy(np.stack(b, axis=0))
        len_org = torch.from_numpy(np.stack(c, axis=0))
        sp_id = torch.from_numpy(np.stack(d, axis=0))

        return {'melsp': melsp, 'pitch': pitch, 'len_org': len_org, 'sp_id': sp_id}

class MultiSampler(Sampler):
    def __init__(self, num_samples, n_repeats, shuffle=False):
        self.num_samples = num_samples # 2
        self.n_repeats = n_repeats # 1
        self.shuffle = shuffle

    def gen_sample_array(self):
        arr = torch.arange(self.num_samples, dtype=torch.int64) # tensor([0, 1])
        self.sample_idx_array = arr.repeat(self.n_repeats) # tensor([0, 1])
        if self.shuffle:
            randperm = torch.randperm(len(self.sample_idx_array)) # tensor([0, 1])
            self.sample_idx_array = self.sample_idx_array[randperm] # tensor([0, 1])
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.sample_idx_array)

def get_loader(opt):
    dataset_path = '/storage/mskim/English_voice/dataset_remove_noise/'
    dataset_train = accent(dataset_path)

    sample = MultiSampler(len(dataset_train), opt.samplier, shuffle=False)
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))

    data_loader = data.DataLoader(dataset=dataset_train, batch_size=opt.batch_size,
                                  sampler=sample,
                                  shuffle=False,
                                  num_workers=opt.num_workers,
                                  drop_last=True,
                                  pin_memory=False,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=MyCollator())

    return data_loader

if __name__ == '__main__':
    from ssun.Voice_trans.config import Config

    config = Config()
    dataload = get_loader(config.opt)
    for batch_id, data in enumerate(dataload, 1):
        print()

    mel_tmp, mfcc_tmp, pitch_tmp, sp_id = accent('/storage/mskim/English_voice/').__getitem__(0)
