
# import packages
import os
import random
import numpy as np
from scipy import stats
from scipy.stats import shapiro, ttest_rel, wilcoxon
import tensorflow as tf
import matplotlib.ticker as ticker

def flatten_list(lst):
    """
    Flatten a list of list
    """
    return [item for sublist in lst for item in sublist]

def seed_tensorflow(seed=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def stat_test(x, y, alternative='two-sided'):
    
    alpha = 0.05
    _, p = shapiro(y - x)
    if p > alpha:
        norm = 1
        stat, pval = ttest_rel(x, y, alternative=alternative)
    else:
        norm = 0
        stat, pval = wilcoxon(x, y, alternative=alternative)
        
    if pval < 0.001:
        note = '***'
    elif pval < 0.01:
        note = '**'
    elif pval < 0.05:
        note = '*'
    else:
        note = ''
    
    return stat, pval, norm, note

def conf_interval(x, mode='ci', dist='t', ci=0.95):
    
    mean = np.mean(x, axis=0)
    sem = stats.sem(x, axis=0)
    if mode=='ci':
        if dist == 't':
            lower, upper = stats.t.interval(ci, x.shape[0]-1, loc=mean, scale=sem)
        elif dist == 'normal':
            lower, upper = stats.norm.interval(ci, loc=mean, scale=sem) 
    else: # mode=='sem':
        lower = mean - sem
        upper = mean + sem        
    return mean, lower, upper

def list2str(lst, sep=','):
    """
    Convert list into string
    """
    return sep.join(str(item) for item in lst)

def str2list(s, sep=',', dtype=str):
    """
    Convert list into string
    """
    return list(map(dtype, s.split(sep)))



def count_params(model):
    trainable_count = tf.reduce_sum(
        [tf.reduce_prod(tf.cast(v.shape, tf.int32)) for v in model.trainable_variables]).numpy().astype(int)
    non_trainable_count = tf.reduce_sum(
        [tf.reduce_prod(tf.cast(v.shape, tf.int32)) for v in model.non_trainable_variables]).numpy().astype(int)
    total_count = trainable_count + non_trainable_count
    print(f"Total params: {total_count}")
    print(f"Trainable params: {trainable_count}")
    print(f"Non-trainable params: {non_trainable_count}")
    return total_count, trainable_count, non_trainable_count

def customize_plot(ax, xlabel, ylabel, xticks, xticklabels, title=None, xlim=None, ylim=None, fontsize=12, rotation=0):
    """
    Customize plot
    """
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=5))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=rotation)
    ax.xaxis.set_tick_params(labelsize=fontsize-1.5, width=1.25)
    ax.yaxis.set_tick_params(labelsize=fontsize-1.5, width=1.25)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+1)
    #ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    for axis in ['bottom','left','top','right']:
        ax.spines[axis].set_linewidth(1.25)

def legend_plot(ax, loc='best', box_xy=None, title=None, fontsize=12, handlelength=1.5, columnspacing=1.5, ncol=1, frameon=False):
    """
    Modify plot legend
    """
    leg = ax.legend(loc=loc, bbox_to_anchor=box_xy, title=title, title_fontsize=fontsize+1, fontsize=fontsize, handlelength=handlelength, ncol=ncol, columnspacing=columnspacing, frameon=frameon)        
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(1.25) 
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)