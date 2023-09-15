import sys,os
sys.path.append("../..")
import numpy as np
import librosa
from librosa.filters import mel
import glob
from scipy import signal

from numpy.random import RandomState
from pysptk import sptk
from sklearn.preprocessing import StandardScaler

def normalize(in_dir, mean, std):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for filename in os.listdir(in_dir):
        filename = os.path.join(in_dir, filename)
        values = (np.load(filename) - mean) / std
        np.save(filename, values)

        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))

    return min_value, max_value

if __name__=='__main__':
    np.load('/storage/mskim/English_voice/make_dataset/new/pitch_state.npy')

    dataset_path = sorted(glob.glob('/storage/mskim/English_voice/train/' + "*/*.wav"))
    path = '/storage/mskim/English_voice/make_dataset/'
    mel_basis = mel(sr=16000, n_fft=1024, fmin=0, fmax=8000, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    fs = 16000
    lo, hi = 50, 600
    b, a = signal.butter(N=5, Wn=30, fs=fs,btype='high')

    os.makedirs((os.path.join(path, "new/make_pitch(only_pitch)")), exist_ok=True)
    pitch_scaler = StandardScaler()

    for index in range(0, len(dataset_path)):
        data = dataset_path[index % len(dataset_path)]
        data_wav, data_sr = librosa.load(data, sr=fs)
        assert data_sr!=None, "sr is None, cheak"

        if data_wav.shape[0] % 256 ==0:
            data_wav=np.concatenate((data_wav, np.array([1e-6])), axis=0)
        # make and save pitch cont
        data_filt = signal.filtfilt(b, a, data_wav)

        seed = RandomState(int(data.split('/')[-1].split('_')[-1][:-4]))
        wav = data_filt * 0.96 + (seed.rand(data_filt.shape[0]) - 0.5) * 1e-06
        pitch = sptk.rapt(wav.astype(np.float32) * 32768, fs, hopsize=256, min=lo, max=hi, otype='pitch')

        if len(pitch>0):
            pitch_scaler.partial_fit(pitch.reshape((-1,1)))
        np.save(path + 'new/make_pitch(only_pitch)/{}_{}'.format(data.split('/')[-2], data.split('/')[-1][:-4]), pitch.astype(np.float32), allow_pickle=False)

    pitch_mean = pitch_scaler.mean_[0]
    pitch_std = pitch_scaler.scale_[0]
    pitch_min, pitch_max = normalize(os.path.join(path, "new/make_pitch(only_pitch)"), pitch_mean, pitch_std)

    stats = {"pitch_min": float(pitch_min), "pitch_max": float(pitch_max), "pitch_mean": float(pitch_mean), "pitch_std": float(pitch_std)}
    np.save(path + 'new/pitch_state.npy',stats)

