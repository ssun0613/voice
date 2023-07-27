import os
import numpy as np
import glob
import torch
from torch.utils import data
from torch.utils.data.sampler import Sampler

LABEL = {'africa': 0, 'australia': 1, 'canada' : 2, 'england' : 3, 'hongkong' : 4, 'us' : 5}

class accent():
    def __init__(self, dataset_path, is_training=True,  mode='speechsplit'):
        self.dataset_dir = dataset_path
        self.is_training = is_training
        self.dataset_mel, self.dataset_mfcc, self.dataset_pitch = self.data_load_npy()
        self.dataset_size = len(self.dataset_mel)

    def data_load_npy(self):
        # dataset_mel = sorted(glob.glob(self.dataset_dir + "make_dataset/make_mel/*.npy"))
        # dataset_mfcc = sorted(glob.glob(self.dataset_dir + "make_dataset/make_mfcc/*.npy"))
        # dataset_pitch = sorted(glob.glob(self.dataset_dir + "make_dataset/make_pitch/*.npy"))
        dataset_mel = sorted(glob.glob(self.dataset_dir + "make_dataset/new/make_mel/*.npy"))
        dataset_mfcc = sorted(glob.glob(self.dataset_dir + "make_dataset/new/make_mfcc/*.npy"))
        dataset_pitch = sorted(glob.glob(self.dataset_dir + "make_dataset/new/make_pitch/*.npy"))

        return dataset_mel, dataset_mfcc, dataset_pitch

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # make_datasave.py reference
        mel_tmp = np.load(self.dataset_mel[index % self.dataset_size])
        mfcc_tmp = np.load(self.dataset_mfcc[index % self.dataset_size])
        pitch_tmp = np.load(self.dataset_pitch[index % self.dataset_size])

        return mel_tmp, mfcc_tmp, pitch_tmp

    def data_label(self, label_path):
        data_label = label_path.split('/')[5]
        data_label_lower = data_label.lower()
        label = np.zeros(len(LABEL))
        label[LABEL[data_label_lower]] = 1.0

        return label

class MyCollator(object):
    def __init__(self):
        self.min_len_seq = 64
        self.max_len_seq = 128
        self.max_len_pad = 192

    def __call__(self, batch):
        # batch[i] is a tuple of __getitem__ outputs
        # batch[i][mel_tmp, mfcc_tmp, pitch_tmp]
        new_batch = []
        for token in batch:
            mel, mfcc, pitch = token
            len_crop = np.random.randint(self.min_len_seq, self.max_len_seq + 1, size=2)  # 1.5s ~ 3s
            # np.random.randint --> self.min_len_seq ~ self.max_len_seq + 1
            left = np.random.randint(0, len(mel) - len_crop[0], size=2)
            # pdb.set_trace()

            mel_crop = mel[left[0]:left[0] + len_crop[0], :] # mel_crop.shape : (len_crop[0], mel_tmp.shape[2])
            mfcc_crop = mfcc[left[0]:left[0] + len_crop[0], :]
            pitch_crop = pitch[left[0]:left[0] + len_crop[0]] # pitch_crop.shape : (len_crop[0],)

            mel_clip = np.clip(mel_crop, 0, 1) # mel_clip.shape : (len_crop[0], mel_tmp.shape[2])
            mfcc_clip = np.clip(mfcc_crop, 0, 1)

            # mel_pad.shape : (self.max_len_pad, mel_tmp.shape[2])
            # pitch_pad.shape : (self.max_len_pad, 1)
            mel_pad = np.pad(mel_clip, ((0, self.max_len_pad - mel_clip.shape[0]), (0, 0)), 'constant')
            mfcc_pad = np.pad(mfcc_clip, ((0, self.max_len_pad - mfcc_clip.shape[0]), (0, 0)), 'constant')
            pitch_pad = np.pad(pitch_crop[:, np.newaxis], ((0, self.max_len_pad - pitch_crop.shape[0]), (0, 0)), 'constant', constant_values=-1e10)

            new_batch.append((mel_pad, mfcc_pad, pitch_pad, len_crop[0]))

        batch = new_batch

        # len(a)=2, a[0].shape : (self.max_len_pad, mel_tmp.shape[2]) | len(b)=2, b[0].shape : (mfcc_tmp.shape[0], mfcc_tmp.shape[1])
        # len(c)=2, c[0].shape : (self.max_len_pad, 1) | len(d)=2, d[0]=len_crop[0]
        a, b, c, d = zip(*batch)

        melsp = torch.from_numpy(np.stack(a, axis=0))
        mfcc = torch.from_numpy(np.stack(b, axis=0))
        pitch = torch.from_numpy(np.stack(c, axis=0))
        len_org = torch.from_numpy(np.stack(d, axis=0))

        return melsp, mfcc, pitch, len_org

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

def get_loader():

    batch_size = 2
    num_workers = 0
    shuffle = True
    samplier = 1

    dataset_path = "/storage/mskim/English_voice/"
    dataset_train = accent(dataset_path)

    sample = MultiSampler(len(dataset_train), samplier, shuffle=shuffle)

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))

    data_loader = data.DataLoader(dataset=dataset_train, batch_size=batch_size,
                                  sampler=sample,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=MyCollator())

    return data_loader

if __name__ == '__main__':
    print("Start data load")

    dataset_path = "/storage/mskim/English_voice/"

    dataset_train = accent(dataset_path)
    mel_tmp, mfcc_tmp, pitch_tmp = dataset_train.__getitem__(6578)



