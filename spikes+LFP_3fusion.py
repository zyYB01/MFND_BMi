"----------3支------------------"
import datetime
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import json
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from bmi.preprocessing import TimeSeriesSplitCustom, transform_data
from bmi.utils import seed_tensorflow, count_params
from bmi.decoders import QRNNDecoder, LSTMDecoder, MLPDecoder
from sklearn.metrics import mean_squared_error
from bmi.metrics import pearson_corrcoef
import time as timer
import os
import pickle
from my_model_MLP import CNN_lstm_v2_lfp_f, CNN_lstm_v2_lfp_ff, CNN_LSTM_SelfAttention, CNN_lstm_v2_lfp_ff_attention
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
    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """

    R2_list = []  # Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]):  # Loop through outputs
        # Compute R2 for each output
        y_mean = np.mean(y_test[:, i])
        R2 = 1 - np.sum((y_test_pred[:, i] - y_test[:, i]) ** 2) / np.sum((y_test[:, i] - y_mean) ** 2)
        R2_list.append(R2)  # Append R2 of this output to the list
    R2_array = np.array(R2_list)
    return R2_array  # Return an array of R2s


def get_bin_lmp(lmp, lmppos):
    # lmppos=lmppos.transpose()
    lmp_bin = []
    lmppos_bin = []
    start_index = 0
    end_index = 17
    # end_index=2
    while (end_index <= lmp.shape[0]):
        if end_index - 1 < 63:  # lmp使用的数据区间长度256ms(256/4=64)
            start_index = start_index + 16
            end_index = start_index + 17
            # start_index=start_index+2
            # end_index=start_index+2
            continue
        start_index_lmp = end_index - 64
        mat1 = lmp[start_index_lmp:end_index, :]
        lmp_bin = lmp_bin + [mat1]
        mat2 = np.mean(lmppos[start_index:end_index, :], axis=0)
        lmppos_bin = lmppos_bin + [mat2]
        start_index = start_index + 16
        end_index = start_index + 17
        # start_index=start_index+2
        # end_index=start_index+2

    lmp_bin = np.array(lmp_bin)
    lmppos_bin = np.array(lmppos_bin)
    return lmp_bin, lmppos_bin


def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿
    print("xls格式表格【追加】写入数据成功！")


def main(args):
    print("Hyperparameter configuration setting")
    if args.config_filepath1:
        # open JSON hyperparameter configuration file
        with open(args.config_filepath1, 'r') as f:
            config = json.load(f)

    else:
        # define model configuration
        config = {'timesteps': args.timesteps,
                  'n_layers': args.n_layers,
                  'units': args.units,
                  'batch_size': args.batch_size,
                  'learning_rate': args.learning_rate,
                  'dropout': args.dropout,
                  'optimizer': args.optimizer,
                  'epochs': args.epochs}
    config['day'] = args.day
    if args.config_filepath2:
        # open JSON hyperparameter configuration file
        with open(args.config_filepath2, 'r') as f:
            config2 = json.load(f)

    else:
        # define model configuration
        config2 = {'timesteps': args.timesteps,
                  'n_layers': args.n_layers,
                  'units': args.units,
                  'batch_size': args.batch_size,
                  'learning_rate': args.learning_rate,
                  'dropout': args.dropout,
                  'optimizer': args.optimizer,
                  'epochs': args.epochs}
    config2['day'] = args.day

    day = config['day']
    folder = "data/LMP_1/"
    files = []
    file_list = os.listdir(folder)
    file_list = sorted(file_list)
    files = day + ".pickle"

    training_range = [0, 0.9]
    valid_range = [0.8, 0.9]
    testing_range = [0.9, 1]
    xlspath = "results/LFPS/test.xls"


    run_start = timer.time()
    begin_day = files
    with open(folder + begin_day, 'rb') as f:
        lmp, lmppos = pickle.load(f,
                                  encoding='latin1')  ##需要得到形状LFP_bin_data:num_trial,764,96    LFP_bin_pos:num_trial,2(只是位置数据)

    # print(f"Reading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()]
        # select the x-y velocity components

    bin_vel = y[:, 2:4]

    y = y[:, 0:2]  # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)

    y_flat = y.reshape(-1, 1)  # 展平为(4165, 1)形状
    # 检查lmppos的每一行是否存在于y中
    rows_in_y = np.isin(lmppos, y_flat).all(axis=1)
    # 获取存在于y中的lmppos元素的索引
    indices = np.where(rows_in_y)[0]
    # 提取相同元素并赋值
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

    print(f"Hyperparameter configuration: {config}")

    # set seed for reproducibility
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

    # specify training set
    X_train = X[train_idx, :]
    # y_train = y[train_idx, :]

    # specify test set
    X_test = X[test_idx, :]
    # y_test = y[test_idx, :]

    X_val = X[valid_set, :]

    X_train_mean = np.nanmean(X_train, axis=0)
    X_train_std = np.nanstd(X_train, axis=0)
    X_train_std = np.square(X_train_std)
    X_train_std = X_train_std + 1e-5
    X_train_std = np.sqrt(X_train_std)

    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std



    # standardize input data
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

    # Zero-center outputs
    y_train_mean = np.mean(y_train, axis=0)
    y_train = y_train - y_train_mean
    y_test = y_test - y_train_mean
    y_val = y_val - y_train_mean


    if args.config_filepath1:
        # open JSON hyperparameter configuration file
        print(f"Using hyperparameter configuration from a file: {args.config_filepath1}")
        with open(args.config_filepath1, 'r') as f:
            config = json.load(f)

    else:
        # define model configuration
        config = {'timesteps': args.timesteps,
                  'n_layers': args.n_layers,
                  'units': args.units,
                  'batch_size': args.batch_size,
                  'learning_rate': args.learning_rate,
                  'dropout': args.dropout,
                  'optimizer': args.optimizer,
                  'epochs': args.epochs}
    config['day'] = args.day

    model_lstm = CNN_lstm_v2_lfp_f(units=config2['units'], dropout=config2['dropout':], num_epochs=config2['epochs'], verbose=1,
                                   batch_size=config2['batch_size'], opt = config['optimizer'], ler =config['learning_rate'])
    history = model_lstm.fit(x_train, y_train, x_val, y_val)
    y_test_predicted_lstm = model_lstm.predict(x_test)

    val_loss = history.history['val_loss']

    v1 = val_loss[-1]  # 最后一个epoch的验证损失



    # Get metric of fit
    R2s_lstm = get_R2(y_test, y_test_predicted_lstm)
    # print('R2s:', R2s_lstm)

    from sklearn import metrics

    RMSEX1 = metrics.mean_squared_error(y_test_predicted_lstm[:, 0], y_test[:, 0]) ** 0.5
    RMSEY1 = metrics.mean_squared_error(y_test_predicted_lstm[:, 1], y_test[:, 1]) ** 0.5

    ccx1 = np.corrcoef(y_test_predicted_lstm[:, 0], y_test[:, 0])
    ccy1 = np.corrcoef(y_test_predicted_lstm[:, 1], y_test[:, 1])
    # cc = np.corrcoef(y_test.transpose((1, 0)), y_test_predicted_lstm.transpose((1, 0)))


    Y_train = y_train
    Y_test = y_test


    model_lstm = CNN_lstm_v2_lfp_ff_attention(units=config['units'], dropout=config['dropout':], num_epochs=200, verbose=1,
                                   batch_size=config['batch_size'])

    history = model_lstm.fit(x_train, X_train, y_train, x_val, X_val, y_val)
    y_test_pred2 = model_lstm.predict(x_test, X_test)

    val_loss = history.history['val_loss']

    v3 = val_loss[-1] # 最后一个epoch的验证损失




    X_train, y_train = transform_data(X_train, y_train, timesteps=config['timesteps'])
    X_test, Y_t = transform_data(X_test, y_test, timesteps=config['timesteps'])

    y_flat = Y_t.reshape(-1, 1)  # 展平为(4165, 1)形状
    # 检查lmppos的每一行是否存在于y中
    rows_in_y = np.isin(y_test, y_flat).all(axis=1)
    # 获取存在于y中的lmppos元素的索引
    indices = np.where(rows_in_y)[0]
    # 提取相同元素并赋值
    y_test = y_test[indices]
    y_test_predicted_lstm = y_test_predicted_lstm[indices]
    y_test_pred2 = y_test_pred2[indices] # 6.0


    # Create and compile model
    print("Compiling and training a model")
    if args.decoder == 'qrnn':
        model = QRNNDecoder(config)

    total_count, _, _ = count_params(model)
    # fit model
    train_start = timer.time()
    history = model.fit(X_train, y_train, validation_data=None, epochs=config['epochs'], verbose=1,
                        callbacks=None)
    train_end = timer.time()
    train_time = (train_end - train_start) / 60
    # print(f"Training the model took {train_time:.2f} minutes")

    # predict using the trained model
    y_test_pred = model.predict(X_test, batch_size=config['batch_size'], verbose=1)

    # evaluate performance
    # print("Evaluating the model performance")
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    cc_test = pearson_corrcoef(y_test, y_test_pred)

    val_loss = history.history['loss']
    v2 = val_loss[-1]  # 最后一个epoch的验证损失




    rmse_test2 = mean_squared_error(y_test, y_test_pred2, squared=False)
    cc_test2 = pearson_corrcoef(y_test, y_test_pred2)



    v = v1+v2+v3

    fy_test_pred2 = (y_test_predicted_lstm * (v-v1) + y_test_pred * (v-v2) + y_test_pred2 * (v-v3))/v

    fcc_test2 = pearson_corrcoef(y_test, fy_test_pred2)

    frmse_test2 = mean_squared_error(y_test, fy_test_pred2, squared=False)

    # print('rmsex1:', RMSEX1)
    # print('rmsey1:', RMSEY1)
    # print('ccx1:', ccx1)
    # print('ccy1:', ccy1)
    #
    cc1 = (ccx1[0, 1] + ccy1[0, 1]) / 2
    RMSE1 = (RMSEX1 + RMSEY1) / 2
    # print(' | RMSE1:', RMSE1, '  cc1:', cc1)
    #
    # rmse_test_folds.append(rmse_test)
    # cc_test_folds.append(cc_test)
    #
    # print(f" | RMSE2 test = {rmse_test_folds[0]:.2f}, CC2 test = {cc_test_folds[0]:.2f}")
    #
    # print(f" | RMSE3 test = {rmse_test2}, CC3 test = {cc_test2}")
    #
    # print("fusion")
    # print(f" |    {day}: fCC test = {fcc_test2}")
    #

    value = [
        [str(day), str(RMSE1), str(cc1), str(rmse_test), str(cc_test), str(rmse_test2), str(cc_test2), str(frmse_test2), str(fcc_test2)]]

    write_excel_xls_append(xlspath, value)




if __name__ == '__main__':

    files_days = ['天数日期'] # files_days = ['indy_20170124_01']


    for day in files_days:
        parser = argparse.ArgumentParser()
        # Hyperparameters
        parser.add_argument('--input_filepath', type=str, default=f"C:/Users/YUBAO/Documents/py_project/spike_bmi-main tensorflow（v2.3.0）/spike_bmi-main/data/dataset/{day}_baks.h5",
                            help='Path to the dataset file')

        parser.add_argument('--seed', type=float, default=42, help='Seed for reproducibility')
        parser.add_argument('--feature', type=str, default='mua', help='Type of spiking activity (sua or mua)')
        parser.add_argument('--decoder', type=str, default='qrnn', help='Deep learning based decoding algorithm')
        parser.add_argument('--config_filepath1', type=str, default=f"C:/Users/YUBAO/Documents/py_project/spike_bmi-main tensorflow（v2.3.0）/spike_bmi-main/params/{day}_mua_baks_qrnn.json",
                            help='JSON hyperparameter configuration file')
        parser.add_argument('--config_filepath2', type=str,
                            default=f"C:/Users/YUBAO/Documents/py_project/spike_bmi-main tensorflow（v2.3.0）/spike_bmi-main/params/{day}_mua_baks_qrnn.json",
                            help='JSON hyperparameter configuration file')
        parser.add_argument('--timesteps', type=int, default=5, help='Number of timesteps')
        parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
        parser.add_argument('--units', type=int, default=600, help='Number of units (hidden state size)')
        parser.add_argument('--window_size', type=int, default=2, help='Window size')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
        parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--metric', type=str, default='mse', help='Predictive performance metric')
        parser.add_argument('--loss', type=str, default='mse', help='Loss function')
        parser.add_argument('--verbose', type=int, default=0, help='Wether or not to print the output')
        parser.add_argument('--day', type=str, default=day, help='Number of days')

        args = parser.parse_args()
        main(args)

