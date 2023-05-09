import os
import sys
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))  # 10^-5
b, a = butter_highpass(30, 16000, order=5)

spk2gen = pickle.load(open('assets/spk2gen.pkl', "rb"))

# Modify as needed
rootDir = 'assets/wavs'
targetDir_f0 = 'assets/raptf0'
targetDir = 'assets/spmel'

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir in sorted(subdirList):
    print(subdir)

    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    if not os.path.exists(os.path.join(targetDir_f0, subdir)):
        os.makedirs(os.path.join(targetDir_f0, subdir))
    _, _, fileList = next(os.walk(os.path.join(dirName, subdir)))

    # if spk2gen[subdir] == 'M':
    #     lo, hi = 50, 250
    # elif spk2gen[subdir] == 'F':
    #     lo, hi = 100, 600
    # else:
    #     raise ValueError
    lo, hi = 50, 600

    prng = RandomState(int(subdir[1:]))
    for fileName in sorted(fileList):
        x, fs = sf.read(os.path.join(dirName, subdir, fileName))
        assert fs == 16000
        if x.shape[0] % 256 == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        y = signal.filtfilt(b, a, x)
        wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06

        D = pySTFT(wav, fft_length=1024, hop_length=256).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum (min_level, D_mel)) - 16
        S = (D_db + 100) / 100

        # extract f0
        f0_rapt = sptk.rapt(wav.astype(np.float32) * 32768, fs, 256, min=lo, max=hi, otype=2)
        index_nonzero = (f0_rapt != -1e10)
        mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
        f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

        assert len(S) == len(f0_rapt)

        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)
        np.save(os.path.join(targetDir_f0, subdir, fileName[:-4]),
                f0_norm.astype(np.float32), allow_pickle=False)


#   https://github.com/r9y9/pysptk/blob/master/pysptk/sptk.py
#   def rapt(x, fs, hopsize, min=60, max=240, voice_bias=0.0, otype="f0"):
#     """RAPT - a robust algorithm for pitch tracking
#     Parameters
#     ----------
#     x : array, dtype=np.float32
#         A whole audio signal
#     fs : int
#         Sampling frequency.
#     hopsize : int
#         Hop size.
#     min : float, optional
#         Minimum fundamental frequency. Default is 60.0
#     max : float, optional
#         Maximum fundamental frequency. Default is 240.0
#     voice_bias : float, optional
#         Voice/unvoiced threshold. Default is 0.0.
#     otype : str or int, optional
#         Output format
#             (0) pitch
#             (1) f0
#             (2) log(f0)
#         Default is f0.
#     Notes
#     -----
#     It is assumed that input array ``x`` has np.float32 dtype, while swipe
#     assumes np.float64 dtype.
#     Returns
#     -------
#     f0  : array, shape(``np.ceil(float(len(x))/hopsize)``)
#         Estimated f0 trajectory
