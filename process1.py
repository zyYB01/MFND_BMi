import os
import pickle
import h5py
import numpy as np
from scipy.io import savemat
from scipy import signal
from scipy.fftpack import fft

def find_index(end_t, lfpt, search_start):
    for i in range(search_start, lfpt.shape[0]):
        if lfpt[i] >= end_t:
            return i

def get_bin_LFP(lfpdata, lfpt, cursor_pos, pos_t):
    LFP_bin_data = []
    LFP_bin_pos = []
    start_pos = 0
    end_pos = 17
    search_start = 0
    while end_pos <= (cursor_pos.shape[1] - 1):
        end_t = pos_t[0, end_pos - 1]
        end_lfp = find_index(end_t, lfpt, search_start)
        if end_lfp < 256:
            start_pos = start_pos + 16
            end_pos = start_pos + 17
            search_start = end_lfp
            continue
        if end_lfp + 1 > lfpt.shape[0]:
            break
        lfp_mat = lfpdata[end_lfp - 255:end_lfp + 1, :]
        LFP_bin_data = LFP_bin_data + [lfp_mat]
        pos_mat = np.mean(cursor_pos[:, start_pos:end_pos], axis=1)
        LFP_bin_pos = LFP_bin_pos + [pos_mat]
        start_pos = start_pos + 16
        end_pos = start_pos + 17
        search_start = end_lfp
    return np.array(LFP_bin_data), np.array(LFP_bin_pos)

def meanfilter(lfpdata, lfpt):
    lmp = []
    lmpt = []
    numpoint = lfpdata.shape[0]
    startlmp = 0
    endlmp = 256
    startt = endlmp - 256
    while (endlmp <= numpoint):
        mat1 = lfpdata[startlmp:endlmp, :]
        mat2 = lfpt[startt:endlmp]
        mat1_mean = np.mean(mat1, axis=0)
        mat2_mean = np.mean(mat2)
        lmp = lmp + [mat1_mean]
        lmpt = lmpt + [mat2_mean]
        startlmp = startlmp + 4
        endlmp = startlmp + 256
        startt = endlmp - 256
    lmp = np.array(lmp)
    lmpt = np.array(lmpt)
    return lmp, lmpt

def align_lmp_cursor(lmp1, lmpt1, cursor_pos_filter, pos_t):
    pos_t = np.squeeze(pos_t)
    if (lmpt1[0] > pos_t[0]) or (lmpt1[-1] < pos_t[-1]):
        print("error! time is not alignmented")
        return [], []
    length1 = len(lmpt1)
    length2 = len(pos_t)
    for i in range(1, length1):
        if (pos_t[0] <= lmpt1[i]) and (pos_t[0] >= lmpt1[i - 1]):
            print(i)
            break
    if (pos_t[0] - lmpt1[i - 1]) < (lmpt1[i] - pos_t[0]):
        start_index = i - 1
    else:
        start_index = i
    lmp2 = lmp1[start_index:start_index + length2, :]

    return lmp2, cursor_pos_filter

folder_LFP = "data/LFP_1/"
folder_spike = "data/raw/"
result_folder = "data/LMP_1/"

files = []
file_list = os.listdir(folder_LFP)
for file in file_list:
    if file.endswith(".pickle"):
        files.append(file)

for file in files:
    print("正在处理" + file)
    filename_nosuffix = file[:-7]
    with open(folder_LFP + file, 'rb') as f:
        lfpdata, lfpt = pickle.load(f, encoding='latin1')
        num_chan = lfpdata.shape[1]
        if num_chan != 96:
            print('error：通道数为')
            print(num_chan)
    data = h5py.File(folder_spike + filename_nosuffix + '.mat')
    cursor_pos = data['cursor_pos'][:]
    pos_t = data['t'][:]
    cursor_pos_filter = cursor_pos

    [lmp1, lmpt1] = meanfilter(lfpdata, lfpt)
    [lmp2, lmppos] = align_lmp_cursor(lmp1, lmpt1, cursor_pos_filter, pos_t)

    with open(result_folder + filename_nosuffix + '.pickle', 'wb') as f:
        pickle.dump([lmp2, lmppos], f)
