

import numpy as np
from numpy.core.fromnumeric import mean, squeeze
from sklearn.metrics import mean_squared_error

def pearson_corrcoef(ytrue, ypred, multioutput="uniform_average"):
   
    assert ytrue.shape == ypred.shape, "both data must have same shape"
    if ytrue.ndim == 1:
        ytrue = np.expand_dims(ytrue, axis=1)
        ypred = np.expand_dims(ypred, axis=1)

    pearson_score = []
    for i in range(ytrue.shape[1]): # Loop through outputs
        score = np.corrcoef(ytrue[:,i], ypred[:,i], rowvar=False)[0,1] # choose the cross-covariance
        pearson_score.append(score)
    pearson_score = np.asarray(pearson_score)
    if multioutput == 'raw_values':
        return pearson_score
    elif multioutput == 'uniform_average':
        return np.average(pearson_score)

def normalized_mse(ytrue, ypred, multioutput='uniform_average', squared=True, norm='minmax'):
   
    error = mean_squared_error(ytrue, ypred, multioutput=multioutput, squared=squared)
    if norm == 'minmax':
        norm_error = error / (ytrue.max() - ytrue.min())
    elif norm == 'mean':
        norm_error = error / ytrue.mean()
    elif norm == 'std':
        norm_error = error / ytrue.std()
    return norm_error
