import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import glob

class accent():
    def __init__(self, dataset_path, is_training=True, mode='speechsplit'):
        self.dataset_dir = dataset_path
        self.is_training = is_training
        self.mode = mode
        self.data_path = self.data_load()
        self.dataset_size = len(self.data_path)

    def data_load(self):
        data = sorted(glob.glob(self.dataset_dir + "*/*.wav"))
        return data

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        data_wav, data_sr = librosa.load(self.data_path[index % self.dataset_size], sr=16000)
        lable = self.data_lable(self.data_path[index % self.dataset_size])

        # if self.mode =='speechsplit':
        #     data_source, data_target = self.source_target(data_wav)
        #     return data_source, data_target, lable

        data_voice_mel = librosa.feature.melspectrogram(data_wav, sr=16000)
        data_voice_mel_s = librosa.power_to_db(data_voice_mel, ref=np.max)
        data_voice_s = librosa.feature.mfcc(data_voice_mel_s, sr=16000, n_mfcc=20, n_fft=400, hop_length=260)
        data_voice = librosa.feature.mfcc(data_wav, sr=16000, n_mfcc=20,n_fft=400, hop_length=260)

        return data_voice_s, data_voice

    def source_target(self, data_wav):
        data_source, data_target = librosa.feature.mfcc(data_wav)
        return data_source, data_target

    def data_lable(self, lable_path):
        data_lable = lable_path.split('/')[5]

        if data_lable == 'africa':
            lable = [1, 0, 0, 0, 0, 0]
        elif data_lable == 'australia':
            lable = [0, 1, 0, 0, 0, 0]
        elif data_lable == 'canada':
            lable = [0, 0, 1, 0, 0, 0]
        elif data_lable == 'england':
            lable = [0, 0, 0, 1, 0, 0]
        elif data_lable == 'hongkong':
            lable = [0, 0, 0, 0, 1, 0]
        elif data_lable == 'us':
            lable = [0, 0, 0, 0, 0, 1]

        return lable

    # def mode(self):
    # return data_source, data_target, lable


if __name__ == '__main__':
    print("Start data load")

    dataset_path = "/storage/mskim/English_voice/train/"
    dataset_train = accent(dataset_path)
    data_voice_s, data_voice = dataset_train.__getitem__(4567)

    plt.figure(figsize=(12,4))
    librosa.display.specshow(data_voice_s)
    # librosa.display.specshow(data_voice)
    plt.ylabel('MFCC coeffs')
    plt.xlabel('Time')
    plt.title('MFCC')

    plt.colorbar()
    plt.tight_layout()





