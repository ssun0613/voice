import sys
sys.path.append("..")
import numpy as np
import librosa
from librosa.filters import mel
import glob
from scipy import signal
from numpy.random import RandomState
from pysptk import sptk
import scipy.fftpack as fft
from sklearn.preprocessing import minmax_scale

# path = "/storage/mskim/English_voice/train/"
# dataset_path = sorted(glob.glob(path + "*/*.wav"))

def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    # index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0

def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    fft_window = signal.get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)

if __name__=='__main__':

    dataset_path = sorted(glob.glob('/storage/mskim/English_voice/train/' + "*/*.wav"))
    path = '/storage/mskim/English_voice/make_dataset/'
    mel_basis = mel(sr=16000, n_fft=1024, fmin=0, fmax=8000, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    fs = 16000
    lo, hi = 50, 600
    b, a = signal.butter(N=5, Wn=30, fs=fs,btype='high')

    for index in range(0,len(dataset_path)):
        data = dataset_path[index % len(dataset_path)]
        data_wav, data_sr = librosa.load(data, sr=fs)
        assert data_sr!=None, "sr is None, cheak"

        if data_wav.shape[0] % 256 ==0:
            data_wav=np.concatenate((data_wav, np.array([1e-6])), axis=0)

        # # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # # make and save mel-spectrogram
        # data_voice_mel = librosa.feature.melspectrogram(y=data_wav, sr=data_sr, hop_length=160, n_mels=512)
        # np.save(path + 'make_mel/{}_{}'.format(data.split('/')[-2], data.split('/')[-1][:-4]), data_voice_mel.astype(np.float32), allow_pickle=False)
        #
        # # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # # make and save mfcc
        # data_voice_mel_s = librosa.power_to_db(data_voice_mel, ref=np.max)
        # data_voice = librosa.feature.mfcc(S=data_voice_mel_s, sr=data_sr, hop_length=160, n_mfcc=100, n_fft=400)
        # np.save(path + 'make_mfcc/{}_{}'.format(data.split('/')[-2], data.split('/')[-1][:-4]), data_voice.astype(np.float32), allow_pickle=False)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # make and save pitch cont
        data_filt = signal.filtfilt(b, a, data_wav)

        seed = RandomState(int(data.split('/')[-1].split('_')[-1][:-4]))
        wav = data_filt * 0.96 + (seed.rand(data_filt.shape[0]) - 0.5) * 1e-06

        D = pySTFT(wav, fft_length=1024, hop_length=256).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = (D_db + 100) / 100

        D_log = 20 * np.log10(D_mel.T)
        mfcc = fft.dct(D_log, axis=0, norm='ortho').T
        mfcc[:, 0:4] = 0
        mfcc = minmax_scale(mfcc, axis=1)

        pitch = sptk.rapt(wav.astype(np.float32) * 32768, fs, hopsize=256, min=lo, max=hi, otype='pitch')
        index_nonzero = (pitch != -1e10)
        pitch_mean, pitch_std = np.mean(pitch[index_nonzero]), np.std(pitch[index_nonzero])
        pitch_norm = speaker_normalization(pitch, index_nonzero, pitch_mean, pitch_std)

        np.save(path + 'new/make_mel/{}_{}'.format(data.split('/')[-2], data.split('/')[-1][:-4]), S.astype(np.float32), allow_pickle=False)
        np.save(path + 'new/make_mfcc/{}_{}'.format(data.split('/')[-2], data.split('/')[-1][:-4]), mfcc.astype(np.float32), allow_pickle=False)
        np.save(path + 'new/make_pitch/{}_{}'.format(data.split('/')[-2], data.split('/')[-1][:-4]), pitch_norm.astype(np.float32), allow_pickle=False)

        if index % 500 == 0:
            print("\n~ing")