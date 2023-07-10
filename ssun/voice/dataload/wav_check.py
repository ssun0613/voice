import wave
import numpy as np
import pandas as pd
import glob

if __name__=='__main__':
    dataset_path = "/storage/mskim/English_voice/train/"
    data = sorted(glob.glob(dataset_path + "*/*.wav"))
    frequency = []
    size = []
    channels = []
    samples = []

    for i in range(len(data)):

        wav =wave.open(data[i])

        sampling_frequency = wav.getframerate()
        sampling_size = wav.getsampwidth()
        num_channels = wav.getnchannels()
        num_samples = wav.getnframes()

        frequency.append(sampling_frequency)
        size.append(sampling_size)
        channels.append(num_channels)
        samples.append(num_samples)

        # print("Sampling frequency : %d [Hz]" %sampling_frequency)
        # print("Sampling size : %d [Byte]" % sampling_size)
        # print("Number of channels : %d" % num_channels)
        # print("Number of samples : %d\n" % num_samples)

    df = pd.DataFrame({"sampling_frequency":frequency, "sampling_size":size, "num_channels":channels, "num_samples": samples})
    df.to_csv('/storage/mskim/English_voice/wav_check/wav_check.csv')
