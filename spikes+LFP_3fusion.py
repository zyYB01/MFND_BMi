
import datetime
import os
import argparse
import json
import h5py

from sklearn.preprocessing import StandardScaler
from bmi.preprocessing import transform_data
from bmi.utils import seed_tensorflow, count_params
from bmi.decoders import QRNNDecoder, LSTMDecoder, MLPDecoder
from sklearn.metrics import mean_squared_error
from bmi.metrics import pearson_corrcoef
import time as timer
import os
import pickle
from my_model_MLP import CNN_lstm_v2_lfp_f, CNN_lstm_v2_lfp_ff_attention
import numpy as np

import xlrd
from xlutils.copy import copy


def get_posvelacc_mat(bin_pos, dt):
    temp_vel = np.diff(bin_pos, axis=0) / dt
    vels_binned = np.concatenate((temp_vel, temp_vel[-1:, :]), axis=0)
    temp_acc = np.diff(vels_binned, axis=0)
    acc_binned = np.concatenate((temp_acc, temp_acc[-1:, :]), axis=0)
    bin_pos = np.concatenate((bin_pos, vels_binned, acc_binned), axis=1)
    return bin_pos


def get_R2(y_test, y_test_pred):
    

    R2_list = []
    for i in range(y_test.shape[1]):
        y_mean = np.mean(y_test[:, i])
        R2 = 1 - np.sum((y_test_pred[:, i] - y_test[:, i]) ** 2) / np.sum((y_test[:, i] - y_mean) ** 2)
        R2_list.append(R2)
    R2_array = np.array(R2_list)
    return R2_array


def get_bin_lmp(lmp, lmppos):
    lmp_bin = []
    lmppos_bin = []
    start_index = 0
    end_index = 64
    while (end_index <= lmp.shape[0]):
        start_index_lmp = end_index - 64
        mat1 = lmp[start_index_lmp:end_index, :]
        lmp_bin = lmp_bin + [mat1]
        mat2 = np.mean(lmppos[start_index_lmp:end_index, :], axis=0)
        lmppos_bin = lmppos_bin + [mat2]
        start_index = start_index + 1
        end_index = end_index + 1

    lmp_bin = np.array(lmp_bin)
    lmppos_bin = np.array(lmppos_bin)
    return lmp_bin, lmppos_bin


def write_excel_xls_append(path, value):
    index = len(value)
    workbook = xlrd.open_workbook(path)
    sheets = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheets[0])
    rows_old = worksheet.nrows
    new_workbook = copy(workbook)
    new_worksheet = new_workbook.get_sheet(0)
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])
    new_workbook.save(path)
    print("xls格式表格【追加】写入数据成功！")


def main(args):
    print("Hyperparameter configuration setting")
    
    # config_file = args.config_file
    config = {}
    
    # if os.path.exists(config_file):
    #     print(f"Reading configuration from {config_file}")
    #     with open(config_file, 'r') as f:
    #         config = json.load(f)
    # else:
    #     print(f"Configuration file {config_file} not found, using default values")
    
    default_config = {
        'timesteps': 4,
        'n_layers_lstm': 2,
        'n_layers_qrnn': 3,
        'units': 150,
        'batch_size': 128,
        'learning_rate': 0.001,
        'dropout': 0.5,
        'optimizer': 'Adam',
        'epochs': 50,
        'day': args.day
    }
    
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    if args.decoder == 'lstm':
        config['n_layers'] = config['n_layers_lstm']
    elif args.decoder == 'qrnn':
        config['n_layers'] = config['n_layers_qrnn']
    else:
        config['n_layers'] = 1
    
    args.timesteps = config['timesteps']
    args.n_layers = config['n_layers']
    args.units = config['units']
    args.batch_size = config['batch_size']
    args.learning_rate = config['learning_rate']
    args.dropout = config['dropout']
    args.optimizer = config['optimizer']
    args.epochs = config['epochs']

    day = config['day']
    folder = "data/LMP_1/"
    files = []
    file_list = os.listdir(folder)
    file_list = sorted(file_list)
    files = day + ".pickle"

    training_range = [0, 0.8]
    valid_range = [0.8, 0.9]
    testing_range = [0.9, 1]
    xlspath = "results/LFPS/test.xls"


    run_start = timer.time()
    begin_day = files
    with open(folder + begin_day, 'rb') as f:
        lmp, lmppos = pickle.load(f,
                                  encoding='latin1')

    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()]

    bin_vel = y[:, 2:4]

    y = y[:, 0:2]

    y_flat = y.reshape(-1, 1)
    rows_in_y = np.isin(lmppos, y_flat).all(axis=1)
    indices = np.where(rows_in_y)[0]
    lmppos = lmppos[indices]
    lmp = lmp[indices]
    lmp_f = np.concatenate((lmp, X), axis=1)
    lmp_f = lmp
    X2 = np.concatenate((X, lmp), axis=1)

    y = bin_vel







    '      ########################################################################################    '

    config['input_dim'] = X.shape[-1]
    config['output_dim'] = y.shape[-1]
    config['window_size'] = args.window_size
    config['loss'] = args.loss
    config['metric'] = args.metric
    config['decoder'] = args.decoder
    config['feature'] = args.feature

 

    seed_tensorflow(args.seed)

    rmse_test_folds = []
    cc_test_folds = []

    num_examples = X.shape[0]
    train_idx = np.arange(np.int64(np.round(training_range[0] * num_examples)),
                          np.int64(np.round(training_range[1] * num_examples)))
    test_idx = np.arange(np.int64(np.round(testing_range[0] * num_examples)) + 11,
                         np.int64(np.round(testing_range[1] * num_examples)))
    valid_set = np.arange(np.int64(np.round(valid_range[0] * num_examples)) + 11,
                          np.int64(np.round(valid_range[1] * num_examples)))

    X_train = X[train_idx, :]

    X_test = X[test_idx, :]

    X_val = X[valid_set, :]



    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)






    multi_bin_neural_data = lmp_f
    num_examples = multi_bin_neural_data.shape[0]

    training_set = np.arange(np.int64(np.round(training_range[0] * num_examples)),
                             np.int64(np.round(training_range[1] * num_examples)))
    testing_set = np.arange(np.int64(np.round(testing_range[0] * num_examples)) + 11,
                            np.int64(np.round(testing_range[1] * num_examples)))
    valid_set = np.arange(np.int64(np.round(valid_range[0] * num_examples)) + 11,
                          np.int64(np.round(valid_range[1] * num_examples)))



    x_train = multi_bin_neural_data[training_set, :]
    y_train = bin_vel[training_set, :]

    x_test = multi_bin_neural_data[testing_set, :]
    y_test = bin_vel[testing_set, :]

    x_val = multi_bin_neural_data[valid_set, :]
    y_val = bin_vel[valid_set, :]

    x_train_mean = np.nanmean(x_train, axis=0)
    x_train_std = np.nanstd(x_train, axis=0)
    x_train_std = np.square(x_train_std)
    x_train_std = x_train_std + 1e-5
    x_train_std = np.sqrt(x_train_std)

    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    x_val = (x_val - x_train_mean) / x_train_std

    y_train_mean = np.mean(y_train, axis=0)
    y_train = y_train - y_train_mean
    y_test = y_test - y_train_mean
    y_val = y_val - y_train_mean



    num_epochs = config['epochs']
    print(f"统一训练轮次: {num_epochs}")

    model1 = CNN_lstm_v2_lfp_f(units=config['units'], dropout=config['dropout'], 
                               num_epochs=1, verbose=1, batch_size=config['batch_size'])



    model2 = CNN_lstm_v2_lfp_ff_attention(units=config['units'], dropout=config['dropout'], 
                                          num_epochs=1, verbose=1, batch_size=config['batch_size'])

    X_train_model3 = X_train.copy()
    y_train_model3 = y_train.copy()
    X_test_model3 = X_test.copy()
    
    if args.decoder == 'qrnn':
        X_train_model3, y_train_model3 = transform_data(X_train_model3, y_train_model3, timesteps=config['timesteps'])
        X_test_model3, Y_t = transform_data(X_test_model3, y_test, timesteps=config['timesteps'])

        y_flat = Y_t.reshape(-1, 1)
        rows_in_y = np.isin(y_test, y_flat).all(axis=1)
        indices = np.where(rows_in_y)[0]
        y_test = y_test[indices]

    print("Compiling and training a model")
    if args.decoder == 'qrnn':
        model3 = QRNNDecoder(config)
    elif args.decoder == 'lstm':
        model3 = LSTMDecoder(config)
    elif args.decoder == 'mlp':
        model3 = MLPDecoder(config)
    total_count, _, _ = count_params(model3)

    print("开始联合训练三个模型...")
    train_start = timer.time()

    losses1 = []
    losses2 = []
    losses3 = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        print("训练模型1 )...")
        history1 = model1.fit(x_train, y_train, x_val, y_val)
        loss1 = history1.history['loss'][-1]
        val_loss1 = history1.history['val_loss'][-1]
        losses1.append(loss1)
        print(f"模型1 - 训练损失: {loss1:.4f}, 验证损失: {val_loss1:.4f}")
        
        print("训练模型2 ...")
        history2 = model2.fit(x_train, X_train, y_train, x_val, X_val, y_val)
        loss2 = history2.history['loss'][-1]
        val_loss2 = history2.history['val_loss'][-1]
        losses2.append(loss2)
        print(f"模型2 - 训练损失: {loss2:.4f}, 验证损失: {val_loss2:.4f}")
        
        print("训练模型3 ...")
        if (args.decoder == 'qrnn') or (args.decoder == 'lstm'):
            history3 = model3.fit(X_train_model3, y_train_model3, validation_data=None, epochs=1, verbose=1, callbacks=None)
        else:
            history3 = model3.fit(X_train_model3, y_train_model3, validation_data=None, epochs=1, verbose=1, callbacks=None)
        loss3 = history3.history['loss'][-1]
        losses3.append(loss3)
        print(f"模型3 - 训练损失: {loss3:.4f}")
        
        epsilon = 1e-10
        total_loss = loss1 + loss2 + loss3 + epsilon
        weight1 = (total_loss - loss1) / total_loss
        weight2 = (total_loss - loss2) / total_loss
        weight3 = (total_loss - loss3) / total_loss
        
        weight_sum = weight1 + weight2 + weight3
        weight1 /= weight_sum
        weight2 /= weight_sum
        weight3 /= weight_sum
        
        print(f"当前权重 - 模型1: {weight1:.4f}, 模型2: {weight2:.4f}, 模型3: {weight3:.4f}")

    train_end = timer.time()
    train_time = (train_end - train_start) / 60
    # print(f"\n联合训练完成，耗时: {train_time:.2f}分钟")

    print("\n开始预测...")
    y_test_predicted_lstm = model1.predict(x_test)
    y_test_pred2 = model2.predict(x_test, X_test)
    
    if (args.decoder == 'qrnn') or (args.decoder == 'lstm'):
        valid_indices = [i for i in indices if i < len(X_test_model3)]
        if valid_indices:
            X_test_model3 = X_test_model3[valid_indices]
            y_test_pred = model3.predict(X_test_model3, batch_size=config['batch_size'], verbose=1)
            y_test_predicted_lstm = y_test_predicted_lstm[valid_indices]
            y_test_pred2 = y_test_pred2[valid_indices]
            y_test = y_test[valid_indices]
        else:
            y_test_pred = model3.predict(X_test_model3, batch_size=config['batch_size'], verbose=1)
    else:
        y_test_pred = model3.predict(X_test_model3, batch_size=config['batch_size'], verbose=1)

    from sklearn import metrics
    RMSEX1 = metrics.mean_squared_error(y_test_predicted_lstm[:, 0], y_test[:, 0]) ** 0.5
    RMSEY1 = metrics.mean_squared_error(y_test_predicted_lstm[:, 1], y_test[:, 1]) ** 0.5

    ccx1 = np.corrcoef(y_test_predicted_lstm[:, 0], y_test[:, 0])
    ccy1 = np.corrcoef(y_test_predicted_lstm[:, 1], y_test[:, 1])

    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    cc_test = pearson_corrcoef(y_test, y_test_pred)
    rmse_test2 = mean_squared_error(y_test, y_test_pred2, squared=False)
    cc_test2 = pearson_corrcoef(y_test, y_test_pred2)

    v1 = history1.history['val_loss'][-1]
    v2 = history3.history['loss'][-1]
    v3 = history2.history['val_loss'][-1]




    config['input_dim'] = X2.shape[-1]
    config['output_dim'] = y.shape[-1]
    config['window_size'] = args.window_size
    config['loss'] = args.loss
    config['metric'] = args.metric
    config['decoder'] = args.decoder
    config['feature'] = args.feature

    print(f"Hyperparameter configuration: {config}")

    seed_tensorflow(args.seed)



    num_examples = X2.shape[0]
    train_idx = np.arange(np.int64(np.round(training_range[0] * num_examples)),
                          np.int64(np.round(training_range[1] * num_examples)))
    test_idx = np.arange(np.int64(np.round(testing_range[0] * num_examples)) + 11,
                         np.int64(np.round(testing_range[1] * num_examples)))
    valid_set = np.arange(np.int64(np.round(valid_range[0] * num_examples)) + 11,
                          np.int64(np.round(valid_range[1] * num_examples)))

    X_train2 = X2[train_idx, :]

    X_test2 = X2[test_idx, :]

    X_val2 = X2[valid_set, :]



    scaler = StandardScaler()
    X_train2 = scaler.fit_transform(X_train2)
    X_test2 = scaler.transform(X_test2)

    if (args.decoder == 'qrnn') or (args.decoder == 'lstm'):
        min_len_train = min(len(X_train2), len(y_train))
        X_train2 = X_train2[:min_len_train]
        y_train = y_train[:min_len_train]
        
        min_len_test = min(len(X_test2), len(y_test))
        X_test2 = X_test2[:min_len_test]
        y_test = y_test[:min_len_test]
        
        X_train2, y_train = transform_data(X_train2, y_train, timesteps=config['timesteps'])
        X_test2, Y_t = transform_data(X_test2, y_test, timesteps=config['timesteps'])


    v = v1+v2+v3

    fy_test_pred2 = (y_test_predicted_lstm * (v-v1) + y_test_pred * (v-v2) + y_test_pred2 * (v-v3))/v

    fcc_test2 = pearson_corrcoef(y_test, fy_test_pred2)



    print("fusion")
    print(f" |    {day}: fCC test = {fcc_test2}")

    # if day == 'indy_demo':
    #     current_time = datetime.datetime.now()
    #     next = "--n-6.0_fusion"
    #     value = [[str(next), str(current_time)]]
    #     write_excel_xls_append(xlspath, value)

    # day = "6.0__fusion_"+args.decoder + day







if __name__ == '__main__':


    files_days = ['indy_demo', ]





    for day in files_days:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_filepath', type=str, default=f"data/dataset/{day}.h5",
                            help='Path to the dataset file')
        parser.add_argument('--output_filepath', type=str, default=f"results/decoder/{day}_qrnn.h5",
                            help='Path to the result file')

        parser.add_argument('--seed', type=float, default=42, help='Seed for reproducibility')
        parser.add_argument('--feature', type=str, default='mua', help='Type of spiking activity (sua or mua)')
        parser.add_argument('--decoder', type=str, default='qrnn', help='Deep learning based decoding algorithm')
        
        parser.add_argument('--timesteps', type=int, default=4, help='Number of timesteps')
        parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
        parser.add_argument('--units', type=int, default=150, help='Number of units (hidden state size)')
        parser.add_argument('--window_size', type=int, default=2, help='Window size')
        parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
        
        parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--loss', type=str, default='mse', help='Loss function')
        parser.add_argument('--metric', type=str, default='mse', help='Predictive performance metric')
        parser.add_argument('--verbose', type=int, default=0, help='Wether or not to print the output')
        parser.add_argument('--day', type=str, default=day, help='Number of days')

        args = parser.parse_args()
        main(args)