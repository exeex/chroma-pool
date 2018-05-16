import torch
import torch.tensor
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import max_pool2d

# 載入data
data = np.load("data/npy/test.wav.npy")
print(data.shape)  # data.shape = (400,1000)  (頻率軸,時間軸)

# data前處理參數
sr = 22050
n_fft = 4096
max_f_idx = 400

# data的頻率軸只有1~400的idx，不是代表真實頻率
# 這邊是計算在idx=1:400時 分別代表的真實頻率(Hz)
fft_freqs = librosa.core.fft_frequencies(sr, n_fft)
fft_freqs = fft_freqs[:max_f_idx]

# 這邊是計算在midi note從0:128 分別代表的真實頻率(Hz)
chm_freqs = [librosa.core.midi_to_hz(x) for x in range(128)]

# 計算midi頻率是對應到fft哪個idx
# idx應該要是個整數
# 但為求準確，這邊是用內插法求精確值，是個float
# 這邊求出來的idx是某個note的中心點，等下要求note邊界
fft_chm_idxs = []
for chm_freq in chm_freqs:
    try:
        idx = np.where((fft_freqs - chm_freq) > 0)[0][0]
        # interpolation
        low = fft_freqs[idx - 1]
        mid = chm_freq
        high = fft_freqs[idx]
        idx = ((idx - 1) * (high - mid) + idx * (mid - low)) / (high - low)
        fft_chm_idxs.append(idx)
    except IndexError:
        break

# torch.from_numpy(np.zeros(800,200))
data = np.load("data/npy/test.wav.npy")

# 秀原圖
# plt.imshow(data)

# 求一個音上下的頻率範圍
for i in range(len(fft_chm_idxs) - 1):
    mid = (fft_chm_idxs[i] + fft_chm_idxs[i + 1]) / 2
    # 次元切割刀
    # plt.axhline(mid, linewidth=2, color='red')

bd_idx = []
for i in range(len(fft_chm_idxs) - 1):
    mid = (fft_chm_idxs[i] + fft_chm_idxs[i + 1]) / 2 - 0.5
    mid = np.rint(mid) + 0.5
    bd_idx.append(mid)
    # 老黃刀法
    # plt.axhline(mid, linewidth=1, color='yellow')
plt.show()

pool_sizes = [np.diff(bd_idx)[:40].sum()] + list(np.diff(bd_idx)[40:])
pool_sizes = [int(x) for x in pool_sizes]
pool_sizes.append(24)
t = torch.from_numpy(data[np.newaxis, :, :])

#########################
# chroma pool           #
#########################

tensors = torch.split(t, pool_sizes, dim=1)

new_tensors = []
for i in range(len(tensors)):
    tensor = tensors[i]
    pool_size = (pool_sizes[i], 3)
    new_tensors.append(max_pool2d(tensor, pool_size))

ret = torch.cat(new_tensors, 1)

# show result of chroma pool
plt.imshow(ret.data[0,:,:])
plt.show()
