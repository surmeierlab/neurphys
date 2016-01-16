#!/usr/bin/env python
__author__ = "Chad Estep (chadestep@gmail.com)"

""" Functions for analyzing oscillatory activity """

import numpy as np
import scipy as sp
from scipy.signal import periodogram
from scipy.stats import gaussian_kde
import pandas as pd

def create_epoch(df, window, step):
    """
    Takes a Pandas DataFrame and blocks it into epochs based on your
    input parameters.

    Parameters
    ----------
    df:
        Pandas Dataframe from 'read_abf' function.
    window: int
        Epoch size based on array index.
    step: int
        Start-to-start number of rows between captured windows (may
        overlap with other windows).

    Returns
    -------
    Multiindexed Pandas DataFrame with column names unchanged and an
    added index level named 'epoch.'

    Notes
    -----
    To keep the funtion as simple as possible, the number of epochs has
    been limited to 999, but if you really need more then feel free to
    change the source code.

    Based on your specified window and step size, your new DataFrame
    may be truncated. 
    """

    window = int(window)
    step = int(step)
    num_rows = len(df.ix[df.index.levels[0][0]])
    num_epochs = int(1 + (num_rows - window) / step)
    sweep_list = []
    sweeps = df.index.levels[0].values
    sweep_names = ['sweep' + str(i + 1).zfill(3) for i in range(len(sweeps))]
    epoch_names = ['epoch' + str(i + 1).zfill(3) for i in range(num_epochs)]
    idx = np.arange(window)
    arrays = [sweep_names,epoch_names,idx]
    index = pd.MultiIndex.from_product(arrays,names=['sweep','epoch',None])
    
    for sweep in sweeps:
        sweep_values = df.ix[sweep].values
        epoch_data = np.array([sweep_values[(0 + step * i):(window + step * i)] for i in range(num_epochs)])
        sweep_data = np.concatenate([epoch_data[i,:,:] for i in range(num_epochs)],axis=0) 
        sweep_list.append(sweep_data)
    concat_sweeps = np.concatenate(sweep_list,axis=0)
    epoch_df = pd.DataFrame(concat_sweeps,columns=df.columns.values)
    epoch_df.set_index(index,inplace=True)
    return epoch_df


def epoch_hist(epoch_df, channel, hist_min, hist_max, num_bins):
    """
    Returns a 1D histogram for each of the epochs created from
    'create_epoch' function.

    Parameters
    ----------
    epoch_df:
        Dataframe from 'create_epoch' function.
    channel: str
        Channel column to be analyzed.
    hist_min: int/float
        Minimum of histogram bin range.
    hist_max: int/float
        Maximum of histogram bin range.
    num_bins: int
        Number of histogram bins.

    Returns
    -------
    Multiindexed Pandas DataFrame with index levels unchanged, but with
    an added column specifiying the bin for each value.

    Notes
    -----
    'bins' column contains the 'leftmost' (smallest?) bin edge.
    """
    hist_arrays = []
    bin_arrays = []
    sweep_names = epoch_df.index.levels[0].values
    epoch_names = epoch_df.index.levels[1].values
    idx = np.arange(num_bins)
    arrays = [sweep_names,epoch_names,idx]
    index = pd.MultiIndex.from_product(arrays,names=['sweep','epoch',None])
    total_epochs = len(sweep_names)*len(epoch_names)
    epoch_size = epoch_df.ix['sweep001'][channel].xs('epoch001').size
    data = epoch_df[channel].values

    for i in range(total_epochs):
        hist, bins = np.histogram(data[(i*epoch_size):((i+1)*epoch_size)],bins=num_bins,range=(hist_min,hist_max))
        hist_arrays.append(hist)
        bin_arrays.append(bins[:num_bins])
    hist_concat = np.concatenate(hist_arrays,axis=0)
    bin_concat = np.concatenate(bin_arrays,axis=0)
    data = list(zip(bin_concat,hist_concat))
    df = pd.DataFrame(data,columns=['bin',channel])
    df.set_index(index, inplace=True)
    return df


def epoch_kde(epoch_df, channel, range_min, range_max, resolution=None):
    """
    Returns a 1D kernel density estimation with automatic bandwidth
    detection for each of the epochs created from the 'create_epoch'
    function.

    Parameters
    ----------
    epoch_df:
        Dataframe from 'create_epoch' function.
    channel: str
        Channel column to be analyzed.
    range_min: int/float
        Minimum of KDE range.
    range_max: int/float
        Maximum of KDE range.
    resolution: int (default: None)
        Determines KDE resolution. >1000 gives very detailed KDEs, but
        the default setting is a great tradeoff with speed. 

    Returns
    -------
    Multiindexed Pandas DataFrame with index levels unchanged, but with
    an added column specifiying the x value for each corresponding
    density value (similar to the 'bin' in the histogram function, but
        not the same)

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.gaussian_kde.html
    """
    
    kde_arrays = []
    x_arrays = []
    if resolution == None:
        resolution = abs(range_min - range_max) * 5
    else:
        resolution = resolution
    sweep_names = epoch_df.index.levels[0].values
    epoch_names = epoch_df.index.levels[1].values
    idx = np.arange(resolution)
    arrays = [sweep_names,epoch_names,idx]
    index = pd.MultiIndex.from_product(arrays,names=['sweep','epoch',None])
    total_epochs = len(sweep_names)*len(epoch_names)
    epoch_size = epoch_df.ix['sweep001'][channel].xs('epoch001').size
    x = np.linspace(range_min, range_max, resolution)
    data = epoch_df[channel].values

    for i in range(total_epochs):
        kde = gaussian_kde(data[(i*epoch_size):((i+1)*epoch_size)])
        kde_data = kde(x)
        kde_arrays.append(kde_data)
        x_arrays.append(x)
    kde_concat = np.concatenate(kde_arrays,axis=0)
    x_concat = np.concatenate(x_arrays,axis=0)
    data = list(zip(x_concat,kde_concat))
    df = pd.DataFrame(data,columns=['x',channel])
    df.set_index(index, inplace=True)
    return df


def epoch_pgram(epoch_df, channel, fs=10e3):
    """
    Returns a periodogram for each of the epochs created from the
    'create_epoch' function.

    Parameters
    ----------
    epoch_df:
        Dataframe from 'create_epoch' function.
    channel: str
        Channel column to be analyzed.
    fs: int (default: 10000)
        Sampling frequency (Hz).

    Returns
    -------
    Multiindexed Pandas DataFrame.

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.periodogram.html
    """
    
    pgram_f_arrays = []
    pgram_den_arrays = []
    fs = int(fs)

    sweep_names = epoch_df.index.levels[0].values
    epoch_names = epoch_df.index.levels[1].values
    total_epochs = len(sweep_names)*len(epoch_names)
    epoch_size = epoch_df.ix['sweep001'][channel].xs('epoch001').size
    idx = (np.arange((epoch_size/2)+1))
    arrays = [sweep_names,epoch_names,idx]
    index = pd.MultiIndex.from_product(arrays,names=['sweep','epoch',None])
    data = epoch_df[channel].values

    for i in range(total_epochs):
        pgram_f, pgram_den = periodogram(data[(i*epoch_size):((i+1)*epoch_size)], fs)
        pgram_f_arrays.append(pgram_f)
        pgram_den_arrays.append(pgram_den)
    pgram_f_concat = np.concatenate(pgram_f_arrays,axis=0)
    pgram_den_concat = np.concatenate(pgram_den_arrays,axis=0)
    data = list(zip(pgram_f_concat,pgram_den_concat))
    df = pd.DataFrame(data,columns=['frequency',channel])
    df.set_index(index, inplace=True)
    return df