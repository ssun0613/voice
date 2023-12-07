import os
import torch
import glob
import numpy as np
from functools import partial
from numpy.random import uniform
from multiprocessing import Process, Manager

from torch.utils import data
from torch.utils.data.sampler import Sampler

LABEL = {'africa': 0, 'australia': 1, 'canada' : 2, 'england' : 3, 'hongkong' : 4, 'us' : 5}

class Utterances(data.Dataset):
    def __init__(self, root_dir, feat_dir, mode):
        self.dataset_dir = '/storage/mskim/English_voice/dataset_remove_noise/'
        self.step = 20
        self.split = 0

        self.dataset_mel, self.dataset_pitch = self.data_load_npy()
        self.dataset_size = len(self.dataset_mel)

    def data_sp_id(self, label_path):
        data_sp_id = label_path.split('/')[-1].split('_')[0]
        data_sp_id_lower = data_sp_id.lower()
        sp_id = np.zeros(len(LABEL))
        sp_id[LABEL[data_sp_id_lower]] = 1.0

        return sp_id

    def data_load_npy(self):
        mel_data=[]
        pitch_data=[]
        dataset_mel = sorted(glob.glob(self.dataset_dir + 'mel/*.npy'))
        dataset_pitch = sorted(glob.glob(self.dataset_dir + 'pitch/*.npy'))

        mel_data.append(dataset_mel[0])
        mel_data.append(dataset_mel[-1])

        pitch_data.append(dataset_pitch[0])
        pitch_data.append(dataset_pitch[-1])

        return mel_data, pitch_data
    def __getitem__(self, index):
        melsp = np.load(self.dataset_mel[index % self.dataset_size])
        f0_org = np.load(self.dataset_pitch[index % self.dataset_size])
        emb_org = self.data_sp_id(self.dataset_mel[index % self.dataset_size])

        return melsp, emb_org, f0_org

    def __len__(self):
        return self.dataset_size

class MyCollator(object):
    def __init__(self, hparams):
        self.min_len_seq = hparams.min_len_seq  # 64
        self.max_len_seq = hparams.max_len_seq  # 128
        self.max_len_pad = hparams.max_len_pad  # 192

    def __call__(self, batch):
        # batch[i] is a tuple of __getitem__ outputs
        new_batch = [] # len(batch[0]) = 3, len(new_batch[0]) = 4
        for token in batch:
            aa, b, c = token # aa=token[0], aa.shape : (18877, 80) | b=token[1], b.shape : (82,) | c=token[2], c.shape : (18877,)
            len_crop = np.random.randint(self.min_len_seq, self.max_len_seq + 1, size=2)  # 1.5s ~ 3s
            # np.random.randint --> self.min_len_seq ~ self.max_len_seq + 1
            left = np.random.randint(0, len(aa) - len_crop[0], size=2)
            # pdb.set_trace()

            a = aa[left[0]:left[0] + len_crop[0], :] # a.shape : (len_crop[0], 80)
            c = c[left[0]:left[0] + len_crop[0]] # c.shape : (len_crop[0],)

            a = np.clip(a, 0, 1) # a.shape : (len_crop[0], 80)

            a_pad = np.pad(a, ((0, self.max_len_pad - a.shape[0]), (0, 0)), 'constant') # a_pad.shape : (self.max_len_pad, 80)
            c_pad = np.pad(c[:, np.newaxis], ((0, self.max_len_pad - c.shape[0]), (0, 0)), 'constant', constant_values=-1e10) # c_pad.shape : (self.max_len_pad, 1)

            new_batch.append((a_pad, b, c_pad, len_crop[0]))

        batch = new_batch

        a, b, c, d = zip(*batch) # len(a)=2, a[0].shape : (self.max_len_pad, 80) | len(b)=2, b[0].shape : (82,) | len(c)=2, c[0].shape : (self.max_len_pad, 1) | len(d)=2, d[0]=len_crop[0]
        melsp = torch.from_numpy(np.stack(a, axis=0)) # np.stack(a, axis=0).shape : (2, self.max_len_pad, 80), melsp.shape : torch.Size([2, self.max_len_pad, 80])
        spk_emb = torch.from_numpy(np.stack(b, axis=0)) # np.stack(b, axis=0).shape : (2, 82), spk_emb.shape : torch.Size([2, 82])
        pitch = torch.from_numpy(np.stack(c, axis=0)) # np.stack(c, axis=0).shape : (2, self.max_len_pad, 1), pitch.shape : torch.Size([2, 192, 1])
        len_org = torch.from_numpy(np.stack(d, axis=0)) # np.stack(d, axis=0).shape : (2, ), len_org.shape : torch.Size([2])

        return melsp, spk_emb, pitch, len_org

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

def get_loader(hparams):

    dataset = Utterances(hparams.root_dir, hparams.feat_dir, hparams.mode)

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))

    data_loader = data.DataLoader(dataset=dataset, batch_size=hparams.batch_size, sampler=MultiSampler(len(dataset), hparams.samplier, shuffle=hparams.shuffle),
                                  num_workers=hparams.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=MyCollator(hparams))
    return data_loader


if __name__ == "__main__":
    from hparams import hparams
    data_load = get_loader(hparams)

    melsp, emb_org, f0_orgd = Utterances(hparams.root_dir, hparams.feat_dir, hparams.mode).__getitem__(0)
