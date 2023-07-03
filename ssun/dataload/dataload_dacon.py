import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import glob



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


class accent_npy():
    def __init__(self, dataset_path, is_training=True, mode='speechsplit'):
        self.dataset_dir = dataset_path
        self.is_training = is_training
        self.mode = mode
        self.dataset_mel, self.dataset_mfcc, self.dataset_pitch = self.data_load_npy()
        self.size_mel, self.size_mfcc, self.size_pitch = len(self.dataset_mel), len( self.dataset_mfcc), len(self.dataset_pitch)

        assert self.size_mel == self.size_mfcc == self.size_pitch, "Error : dataset size"

    def data_load_npy(self):
        dataset_mel = sorted(glob.glob(self.dataset_dir + "make_dataset/make_mel/*.npy"))
        dataset_mfcc = sorted(glob.glob(self.dataset_dir + "make_dataset/make_mfcc/*.npy"))
        dataset_pitch = sorted(glob.glob(self.dataset_dir + "make_dataset/make_pitch/*.npy"))

        return dataset_mel, dataset_mfcc, dataset_pitch

    def __len__(self):
        return self.size_mel, self.size_mfcc, self.size_pitch

    def __getitem__(self, index):
        # make_datasave.py reference
        # if index==0, mel_tmp.shape = (128, 226), mfcc_tmp.shape = (20, 226), pitch_tmp.shape = (452,)
        mel_tmp = np.load(self.dataset_mel[index % self.size_mel])
        mfcc_tmp = np.load(self.dataset_mfcc[index % self.size_mfcc])
        pitch_tmp = np.load(self.dataset_pitch[index % self.size_pitch])

        return data_voice, label

    def data_label(self, label_path):
        data_label = label_path.split('/')[5]
        data_label_lower = data_label.lower()
        label = np.zeros(len(LABEL))
        label[LABEL[data_label_lower]] = 1.0
        return label


if __name__ == '__main__':
    print("Start data load")

    dataset_path = "/storage/mskim/English_voice/"
    dataset_train = accent_npy(dataset_path)
    data_voice, label = dataset_train.__getitem__(4567)

    # plt.figure(figsize=(20,6))
    # plt.subplot(121)
    # librosa.display.specshow(data_voice_s)
    # plt.ylabel('MFCC coeffs')
    # plt.xlabel('Time')
    # plt.title('MFCC : log mel-spectrogram')
    #
    # plt.subplot(122)
    # librosa.display.specshow(data_voice)
    # plt.ylabel('MFCC coeffs')
    # plt.xlabel('Time')
    # plt.title('MFCC : orginal data wav')
    #
    # plt.show()
