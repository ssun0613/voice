# mel_nrw = librosa.amplitude_to_db(np.abs(librosa.stft(data_nrw)), ref=np.max)
# mel_orig = librosa.amplitude_to_db(np.abs(librosa.stft(data_wav)), ref=np.max)
# plt.subplot(2, 1, 1)
# plt.imshow(mel_nrw[::-1, :])
# plt.subplot(2, 1, 2)
# plt.imshow(mel_orig[::-1, :])

import sys,os
sys.path.append("../..")
import numpy as np
import librosa
import glob
from librosa.filters import mel
from scipy import signal
from numpy.random import RandomState
from pysptk import sptk
import scipy.fftpack as fft
from sklearn.preprocessing import minmax_scale
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence
import subprocess
import soundfile as sf


def remove_noise():
    dataset_path = sorted(glob.glob('/storage/mskim/English_voice/train/' + "*/*.wav"))
    path = '/storage/mskim/English_voice/make_dataset/'
    path_1 = '/storage/mskim/English_voice/'
    fs = 22050
    # ------------------------------------------------------------------------------------------

    os.makedirs((os.path.join(path, "noise_remove")), exist_ok=True)
    os.makedirs((os.path.join(path, "noise_split")), exist_ok=True)
    os.makedirs((os.path.join(path_1, "dataset_remove_noise/wav")), exist_ok=True)

    error_path = 0
    error_list = []
    error_messeage = []

    for index in range(0, len(dataset_path)):
        data = dataset_path[index % len(dataset_path)]
        data_wav, data_sr = librosa.load(data, sr=fs)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data_wav)), ref=np.max)
        # ------------------------------------------------------------------------------------------
        sound = AudioSegment.from_file(data)
        # ------------------------------------------------------------------------------------------
        audio_chunks = split_on_silence(sound,
                                        min_silence_len=100,
                                        silence_thresh=-55,
                                        keep_silence=50)
        # ------------------------------------------------------------------------------------------
        data_nrw = 0

        for i, chunk in enumerate(audio_chunks):

            noise_split_path = (os.path.join(path + "noise_split", "{}_"+data.split('/')[-1]).format(i))
            chunk.export(noise_split_path, format='wav')

            input_path = (os.path.join(path + "noise_split", "{}_" + data.split('/')[-1]).format(i))
            noise_remove_path = (os.path.join(path + "noise_remove", "{}_"+data.split('/')[-1]).format(i))

            subprocess.call(['ffmpeg', '-i', input_path, '-af', 'silenceremove=1:0:-60dB', '-y', noise_remove_path])

            data_n_r_w, _ = librosa.load(noise_remove_path)

            if i == 0:
                data_nrw = data_n_r_w
            elif i != 0:
                data_nrw = np.concatenate((data_nrw, data_n_r_w), axis=None)
        try:
            sf.write(path_1 + 'dataset_remove_noise/wav/{}_{}.wav'.format(data.split('/')[-2], data.split('/')[-1][:-4]), data_nrw, samplerate=fs)
            print("{}".format(path_1 + 'dataset_remove_noise/wav/{}_{}'.format(data.split('/')[-2], data.split('/')[-1][:-4])))
            print("{} / {}".format(data_wav.shape, data_nrw.shape))
        except Exception as e:
            error_path = '{}_{}'.format(data.split('/')[-2], data.split('/')[-1][:-4])
            error_list.append(error_path)
            error_messeage.append(e)

        np.save(path_1 + "dataset_remove_noise/error_list_55", error_list)
        np.save(path_1 + "dataset_remove_noise/error_messeage_55", error_messeage)

def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    # index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / (std_f0 + 1e-6) / 4.0
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
    remove_noise()
    path_1 = '/storage/mskim/English_voice/dataset_remove_noise/'
    dataset_path = sorted(glob.glob(path_1 + "wav/*.wav"))
    mel_basis = mel(sr=22050, n_fft=1024, fmin=0, fmax=8000, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    fs = 22050
    lo, hi = 50, 600
    b, a = signal.butter(N=5, Wn=30, fs=fs,btype='high')

    os.makedirs((os.path.join(path_1, "mel")), exist_ok=True)
    os.makedirs((os.path.join(path_1, "mfcc")), exist_ok=True)
    os.makedirs((os.path.join(path_1, "pitch")), exist_ok=True)

    error_list=[]
    error_messeage=[]

    for index in range(0,len(dataset_path)):
        data = dataset_path[index % len(dataset_path)]
        data_wav, data_sr = librosa.load(data, sr=fs)
        assert data_sr!=None, "sr is None, cheak"

        if data_wav.shape[0] % 256 ==0:
            data_wav=np.concatenate((data_wav, np.array([1e-6])), axis=0)

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

        pitch = sptk.rapt(data_wav.astype(np.float32) * 32768, fs, hopsize=256, min=lo, max=hi, otype='pitch')
        index_nonzero = (pitch != -1e10)
        pitch_mean, pitch_std = np.mean(pitch[index_nonzero]), np.std(pitch[index_nonzero])
        pitch_norm = speaker_normalization(pitch, index_nonzero, pitch_mean, pitch_std)

        try:
            np.save(path_1 + 'mel/{}'.format(data.split('/')[-1].split('.')[0]), S.astype(np.float32), allow_pickle=False)
            np.save(path_1 + 'mfcc/{}'.format(data.split('/')[-1].split('.')[0]), mfcc.astype(np.float32), allow_pickle=False)
            np.save(path_1 + 'pitch/{}'.format(data.split('/')[-1].split('.')[0]), pitch_norm.astype(np.float32), allow_pickle=False)
        except Exception as e:
            error_path = '{}'.format(data.split('/')[-1].split('.')[0])
            error_list.append(error_path)
            error_messeage.append(e)


        if index % 500 == 0:
            print("\n~ing")

    np.save(path_1 + "error_list_preprocessing", error_list)
    np.save(path_1 + "error_messeage_preprocessing", error_messeage)