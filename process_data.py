import argparse
import h5py
import numpy as np
from bmi.utils import flatten_list
import numpy as np
from scipy.io import savemat

def main(args):

    num_chan = 96
    print (f"Reading raw data from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        task_pos = f['cursor_pos'][()].T
        target_pos = f['target_pos'][()].T
        task_time = f['t'][()].squeeze()
        spikes = f['spikes'][()].T
        num_unit = spikes.shape[1]
        print(f"Number of channels: {num_chan}, number of units: {num_unit}")
        all_spikes = []
        for i in range(num_chan):
            chan_spikes = []
            for j in range(num_unit):
                if (f[spikes[i,j]].ndim == 2):
                    tmp_spikes = f[spikes[i,j]][()].squeeze(axis=0)
                else:
                    tmp_spikes = np.empty(0)
                chan_spikes.append(tmp_spikes)
            all_spikes.append(chan_spikes)

    sua_trains = []
    len_sua_trains = []    
    for i in range(num_chan):
        for j in range(num_unit):
            if (j > 0) & (all_spikes[i][j].shape[0] > 0):
                sua_train = all_spikes[i][j]
                sua_idx = np.where((sua_train >= task_time[0]) & (sua_train <= task_time[-1]))[0]
                sua_train = sua_train[sua_idx]
                sua_trains.append(sua_train)
                len_sua_trains.append(len(sua_train))      
    num_sua = len(sua_trains)  

    mua_trains = []   
    len_mua_trains = []   
    for i in range(num_chan):
        chan_mua = []
        for j in range(num_unit):
            if (all_spikes[i][j].shape[0] > 0):
                mua_train = all_spikes[i][j]
                mua_idx = np.where((mua_train >= task_time[0]) & (mua_train <= task_time[-1]))[0]
                mua_train = mua_train[mua_idx]
                chan_mua.append(mua_train)
        chan_mua = flatten_list(chan_mua)
        chan_mua.sort()
        mua_trains.append(np.asarray(chan_mua))
        len_mua_trains.append(len(chan_mua))
    num_mua = len(mua_trains) 

    print(f"[Before filtering] Number of SUA: {len(sua_trains)}, Number of MUA: {len(mua_trains)}")

    min_spikerate = 0.5
    task_duration = task_time[-1] - task_time[0]
    min_numspike = int(np.round(min_spikerate * task_duration))

    len_sua_trains = np.asarray(len_sua_trains)
    len_mua_trains = np.asarray(len_mua_trains)

    sua_valid_idx = np.where(len_sua_trains > min_numspike)[0]
    mua_valid_idx = np.where(len_mua_trains > min_numspike)[0]

    sua_trains_valid = []
    for idx in sua_valid_idx:
        sua_trains_valid.append(sua_trains[idx])

    mua_trains_valid = []
    for idx in mua_valid_idx:
        mua_trains_valid.append(mua_trains[idx])

    print(f"[After filtering] Number of SUA: {len(sua_trains_valid)}, Number of MUA: {len(mua_trains_valid)}")

    dt_task = np.diff(task_time).mean()
    task_vel = np.diff(task_pos, axis=0) / dt_task
    task_acc = np.diff(task_vel, axis=0) / dt_task
    task_vel = np.concatenate((task_vel, task_vel[-1:,:]), axis=0)
    task_acc = np.concatenate((task_acc, task_acc[-2:,:]), axis=0)
    task_data = np.concatenate((task_pos, task_vel, task_acc), axis=1) 

    with h5py.File(args.output_filepath, 'w') as f:
        f['task_time'] = task_time
        f['task_data'] = task_data

        dt = h5py.special_dtype(vlen=np.dtype('f8'))
        f.create_dataset('sua_trains', data=np.asarray(sua_trains_valid, dtype=dt))
        f.create_dataset('mua_trains', data=np.asarray(mua_trains_valid, dtype=dt))

    matlab_vars = {}
    matlab_vars[f'task_time'] = task_time
    matlab_vars[f'task_data'] = task_data
    matlab_vars[f'mua_trains'] = np.asarray(mua_trains_valid, dtype=dt)
    savemat('data/spike/', matlab_vars)

    print(f"Finished processing and storing spike and kinematic data : {args.input_filepath}")
    print(f"Finished processing and storing spike and kinematic data into file: {args.output_filepath}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',   type=str,   default=f"data")
    parser.add_argument('--output_filepath',  type=str,    default=f"data/spike/")
    
    args = parser.parse_args()
    main(args)