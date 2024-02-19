import sys,os
sys.path.append("../..")
import numpy as np
import librosa
from librosa.filters import mel
import glob
from scipy import signal

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy.random import RandomState
from pysptk import sptk
from sklearn.preprocessing import StandardScaler


def normalize(in_dir, mean, std):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for filename in os.listdir(in_dir):
        filename = os.path.join(in_dir, filename)
        values = (np.load(filename) - mean) / std
        # np.save(filename, values)values

        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))

    return min_value, max_value

def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    # index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / (std_f0 + 1e-6) / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0

    return f0

def normalize_f0(x):
    # index_nonzero = (x != 0)
    index_nonzero = (x > 0)
    mean_x, std_x = np.mean(x[index_nonzero]), np.std(x[index_nonzero])
    x_norm = speaker_normalization(x, index_nonzero, mean_x, std_x)

    return x_norm

if __name__=='__main__':

    dataset_path = sorted(glob.glob('/storage/mskim/English_voice/train/' + "*/*.wav"))
    path = '/storage/mskim/English_voice/make_dataset/'
    mel_basis = mel(sr=16000, n_fft=1024, fmin=0, fmax=8000, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    fs = 16000
    lo, hi = 50, 600
    hop_length =256
    b, a = signal.butter(N=5, Wn=30, fs=fs,btype='high')

    # os.makedirs((os.path.join(path, "new/make_pitch(only_pitch_rapt)")), exist_ok=True)
    pitch_scaler = StandardScaler()

    for index in range(0, len(dataset_path)):
        data = dataset_path[index % len(dataset_path)]
        data_wav, data_sr = librosa.load(data, sr=fs)

        assert data_sr!=None, "sr is None, cheak"

        # _f0, t = pw.dio(data_wav.astype(np.float64), fs, frame_period=hop_length / fs * 1000)
        # pitch = pw.stonemask(data_wav.astype(np.float64), _f0, t, fs).astype(np.float32)
        #
        # if np.isnan(pitch).any():
        #     print('pitch isnan : {}_{}.npy'.format(data.split('/')[-2], data.split('/')[-1][:-4]))
        #
        # pitch_norm = normalize_f0(pitch)
        #
        # if np.isnan(pitch_norm).any():
        #     print('pitch_norm isnan : {}_{}.npy'.format(data.split('/')[-2], data.split('/')[-1][:-4]))

        pitch = sptk.rapt(data_wav.astype(np.float32) * 32768, fs, hopsize=256, min=lo, max=hi, otype='pitch')
        index_nonzero = (pitch != -1e10)
        pitch_mean, pitch_std = np.mean(pitch[index_nonzero]), np.std(pitch[index_nonzero])
        pitch_norm = speaker_normalization(pitch, index_nonzero, pitch_mean, pitch_std)

        if np.isnan(pitch).any():
            print('pitch isnan : {}_{}.npy'.format(data.split('/')[-2], data.split('/')[-1][:-4]))

        elif np.isnan(pitch_norm).any():
             print('pitch_norm isnan : {}_{}.npy'.format(data.split('/')[-2], data.split('/')[-1][:-4]))

        else:
            np.save(path + 'new/make_pitch(only_pitch_rapt)/{}_{}.npy'.format(data.split('/')[-2], data.split('/')[-1][:-4]), pitch_norm.astype(np.float32), allow_pickle=False)

        if index % 1000 == 0:
            print("save ~ing")

        if len(pitch>0):
            pitch_scaler.partial_fit(pitch.reshape((-1,1)))

    print("\nfinish")

    pitch_mean = pitch_scaler.mean_[0]
    pitch_std = pitch_scaler.scale_[0]
    pitch_min, pitch_max = normalize(os.path.join(path, "new/make_pitch(only_pitch_rapt)"), pitch_mean, pitch_std)

    stats = {"pitch_min": float(pitch_min), "pitch_max": float(pitch_max), "pitch_mean": float(pitch_mean), "pitch_std": float(pitch_std)}
    np.save(path + 'new/pitch_state(only_pitch_rapt).npy',stats)

