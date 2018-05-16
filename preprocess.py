import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import scipy.signal


mp3_folder = "data/wav"
npy_folder = "data/npy"

files = os.listdir(mp3_folder)


def pad_along_axis(array: np.ndarray, target_length, axis=0):

    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b


def preprosess(fortest=False):

    for file in files:

        infile_path =os.path.join(mp3_folder, file)
        outfile_path = os.path.join(npy_folder, file + ".npy")
        print(file)
        try:
            aud, sr = librosa.load(infile_path, sr=None)
            # D = librosa.cqt(aud,sr=sr, n_bins=720, bins_per_octave=120, hop_length=1024)
            D = librosa.stft(aud,n_fft=4096)
            M, P = librosa.magphase(D[:400,1500:2500])
            M = pad_along_axis(M,300,axis=1)
            data = M
            # data = librosa.amplitude_to_db(M)#.astype("float32")
        except Exception as err:
            print("load file %s fail," % infile_path, err)
            continue

        if fortest:
            plt.figure()
            print(data.shape)
            librosa.display.specshow(data, sr=sr)
            plt.show()

            return

        try:
            np.save(outfile_path, data)

        except Exception as err:
            print(err)


if __name__ == "__main__":
    preprosess(fortest=True)

    # a = np.identity(5)
    # b = pad_along_axis(a, 7, axis=1)
    # print(a,a.shape)
    # print(b,b.shape)