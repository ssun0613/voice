import numpy as np
import glob
import torch

if __name__ == '__main__':
    dataset_path = '/storage/mskim/English_voice/'

    dataset_mel = sorted(glob.glob(dataset_path + 'make_dataset/new/make_mel/*.npy'))
    dataset_mfcc = sorted(glob.glob(dataset_path + 'make_dataset/new/make_mfcc/*.npy'))
    dataset_pitch = sorted(glob.glob(dataset_path + 'make_dataset/new/make_pitch(only_pitch_norm)/*.npy'))

    data_nan_mel =[]
    data_nan_mfcc = []
    data_nan_pitch = []
    for i in range(len(dataset_mel)):

        mel_tmp = np.load(dataset_mel[i])
        mfcc_tmp = np.load(dataset_mfcc[i])
        pitch_tmp = np.load(dataset_pitch[i])

        if np.isnan(mel_tmp).any() or np.isnan(mfcc_tmp.any()) or np.isnan(pitch_tmp).any():

            if np.isnan(mel_tmp).any():
                data_nan_mel.append(dataset_mel[i])
                print("dataset_mel isnan : {}".format(dataset_mel[i]))

            elif np.isnan(mfcc_tmp).any():
                data_nan_mfcc.append(dataset_mfcc[i])
                print("dataset_mfcc isnan : {}".format(dataset_mfcc[i]))

            elif np.isnan(pitch_tmp).any():
                data_nan_pitch.append(dataset_pitch[i])
                print("dataset_pitch isnan : {}".format(dataset_pitch[i]))

    print(len(data_nan_mel))
    print(len(data_nan_mfcc))
    print(len(data_nan_pitch))

    if len(data_nan_mel)!=0:
        np.save(dataset_path+'make_dataset/new/data_nan_mel.npy', data_nan_mel)

    elif len(data_nan_mfcc)!=0:
        np.save(dataset_path+'make_dataset/new/data_nan_pitch.npy', data_nan_mfcc)

    elif len(data_nan_pitch)!=0:
        np.save(dataset_path+'make_dataset/new/data_nan_pitch.npy', data_nan_pitch)