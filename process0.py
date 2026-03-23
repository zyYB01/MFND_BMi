from scipy import io
import os
import numpy as np
import h5py
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import math
import pickle
data_file='data/LFP'
data_folder="data/LFP/"
files=[]
file_list=os.listdir(data_file)
for file in file_list:
    if file.endswith(".mat"):
        files.append(file)
for file in files[0:1]:
    print("正在处理"+file)

    file1_path=data_file+'/'+file
    print(file1_path)
    data1=h5py.File(file1_path)
    rawdata=data1['rawdata'][:]
    t=data1['tt'][:]
    rawdata=rawdata.astype(np.float32)

    y=data1['y'][:]
    t2=data1['t'][:]
    y=y.astype(np.float32)
    print(t.shape[0])

    dt_task = np.diff(t2).mean()
    task_vel = np.diff(y, axis=0) / dt_task
    task_vel = np.concatenate((task_vel, task_vel[-1:,:]), axis=0)
    Y = task_vel

    rawaverage=np.average(rawdata,axis=1)
    rawdata=rawdata-rawaverage[:,np.newaxis]

    Fs=1/(t[0,2]-t[0,1])
    ws=300
    wn=ws/(0.5*Fs)
    b,a=signal.butter(4,wn,'highpass')
    num_chan=rawdata.shape[1]
    for i in range(num_chan):
        rawdata[:,i]=signal.filtfilt(b,a,rawdata[:,i])

    rawdata=abs(rawdata)

    ws=12
    wn=ws/(0.5*Fs)
    b, a = signal.butter(1, wn, 'lowpass')
    for i in range(num_chan):
        rawdata[:,i]=signal.filtfilt(b,a,rawdata[:,i])

    num_step=round(Fs/1000)
    num_step=int(num_step)
    num_t=t.shape[1]
    index1=range(0,num_t,num_step)
    new_data=rawdata[index1,:]
    t=t[0,index1]

    filenomat=file[0:-4]
    with open(data_folder+filenomat+'.pickle','wb') as f:
        pickle.dump([new_data,t],f)

    print(file+"处理完毕")
