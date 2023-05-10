import os
import torch
import pickle
import numpy as np

from functools import partial
from numpy.random import uniform
from multiprocessing import Process, Manager # process : 병렬처리하므로써 더 빠르게 결과를 얻을 수 있음. manager : List or Dict 등의 변수를 공유 할 수 있음.

from torch.utils import data
from torch.utils.data.sampler import Sampler

class Utterances(data.Dataset):

    def __init__(self, root_dir, feat_dir, mode):
        self.root_dir = root_dir  # 'assets/spmel'
        self.feat_dir = feat_dir  # 'assets/raptf0'
        self.mode = mode  # 'train'
        self.step = 20
        self.split = 0

        metaname = os.path.join(self.root_dir, "train.pkl")  # 'assets/spmel/train.pkl'
        meta = pickle.load(open(metaname, "rb"))

        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta) * [None])  # <-- can be shared between processes.
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, args=(meta[i:i + self.step], dataset, i, self.mode))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # very important to do dataset = list(dataset)
        if mode == 'train':
            self.train_dataset = list(dataset)
            self.num_tokens = len(self.train_dataset)
        elif mode == 'test':
            self.test_dataset = list(dataset)
            self.num_tokens = len(self.test_dataset)
        else:
            raise ValueError

        print('Finished loading {} dataset...'.format(mode))

    def load_data(self, submeta, dataset, idx_offset, mode):
        for k, sbmt in enumerate(submeta):
            uttrs = len(sbmt) * [None]
            # fill in speaker id and embedding
            uttrs[0] = sbmt[0]  # speaker id
            uttrs[1] = sbmt[1]  # embedding

            # fill in data
            sp_tmp = np.load(os.path.join(self.root_dir, sbmt[2])) # sp_tmp.shape : [0 --> (18877, 80)], [1 --> (18902, 80)]
            f0_tmp = np.load(os.path.join(self.feat_dir, sbmt[2])) # f0_tmp.shape : [0 --> (18877,)], [1 --> (18902,)]

            if self.mode == 'train':
                sp_tmp = sp_tmp[self.split:, :]
                f0_tmp = f0_tmp[self.split:]
            elif self.mode == 'test':
                sp_tmp = sp_tmp[:self.split, :]
                f0_tmp = f0_tmp[:self.split]
            else:
                raise ValueError
            uttrs[2] = (sp_tmp, f0_tmp)
            dataset[idx_offset + k] = uttrs

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset

        list_uttrs = dataset[index]  # len(list_uttrs)=3,
        spk_id_org = list_uttrs[0]  # speaker id, spk_id_org : 'p226'
        emb_org = list_uttrs[1]  # embedding, emb_org.shape : (82,)

        melsp, f0_org = list_uttrs[2]  # melsp.shape : (18877, 80) , f0_org.shape : (18877,)

        return melsp, emb_org, f0_org

    def __len__(self):
        return self.num_tokens


class MyCollator(object):

    def __init__(self, hparams):
        print('start')
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

    my_collator = MyCollator(hparams)

    sampler = MultiSampler(len(dataset), hparams.samplier, shuffle=hparams.shuffle)

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=hparams.batch_size,
                                  sampler=sampler,
                                  num_workers=hparams.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=my_collator)
    return data_loader


if __name__ == "__main__":
    test = Utterances(root_dir='assets/spmel', feat_dir='assets/raptf0', mode='train')
