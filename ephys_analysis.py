__author__ = "Chad Estep (chadestep@gmail.com)"

import numpy as np
import scipy as sp
from scipy.signal import periodogram
from scipy.stats import gaussian_kde
from neo import io
import pandas as pd
# import matplotlib.pyplot as plt

def read_abf(filename):
    """
    Imports ABF file using neo io AxonIO, breaks it down by blocks 
    which are then processed into  a multidimensional pandas dataframe 
    where each block corresponds to a sweep and columns represent time
    and each recorded channel. Channel names can be changed later if 
    necessary.
    
    More documentation necessary.

    Input parameters
    filename  : filename WITH '.abf' extension
    """

    r = io.AxonIO(filename = filename)
    bl = r.read_block(lazy=False, cascade=True)
    num_channels = len(bl.segments[0].analogsignals)
    channels = []
    signals = []
    df_list = []
    sweep_list = []

    for seg_num, seg in enumerate(bl.segments):
        for i in range(num_channels):
            channels.append('channel_' + str(i))
            signals.append(bl.segments[seg_num].analogsignals[i])
        data_dict = dict(zip(channels, signals))
        time = seg.analogsignals[0].times - seg.analogsignals[0].times[0]
        data_dict['time'] = time
        df = pd.DataFrame(data_dict)
        df_list.append(df)
        sweep_list.append('sweep' + str(seg_num + 1).zfill(3))
    return pd.concat(df_list, keys=sweep_list, names=['sweep'])


def create_epoch(df, window, step):
    """
    This function takes an input DataFrame and groups it by its level=0
    index before passing it to the rest of the function to create a new
    Multiindex DataFrame with the original level=0 being the same, and
    an added level=1 being what this function creates.

    To keep things as 'simple' as possible for the other functions, the
    number of epochs has been limited to 999. If you really need more
    than that number, then just go ahead and change the source code.

    NOTE: based on your specified window and step size, your 
    new array may be truncated.

    Input parameters
    df        : input pandas dataframe
    window    : epoch size based on array index
    step      : start-to-start number of rows between captured windows 
             (may overlap with other windows)
    """

    window = int(window)
    step = int(step)
    df_list = []
    sweeps = df.index.levels[0].values
    rows = len(df.ix[df.index.levels[0][0]])
    num_epochs = int(1 + (rows - window) / step)

    for sweep in sweeps:
        for i in range(num_epochs):
            epoch_name = 'epoch' + str(i + 1).zfill(3)
            epoch = df.ix[sweep][(0 + step * i):(window + step * i)]
            arrays = [[sweep]*window,[epoch_name]*window,np.arange(window)]
            index = pd.MultiIndex.from_arrays(arrays, names=['sweep','epoch',None])
            epoch_df = pd.DataFrame(epoch, columns=df.columns.values)
            epoch_df.set_index(index, inplace=True)
            df_list.append(epoch_df)
    return pd.concat(df_list)


def epoch_hist(epoch_df, channel, hist_min, hist_max, num_bins):
    """
    Creates a bunch of 1D histograms of the epochs created from
    ea.rolling_window function.

    Input parameters
    epoch_df  : dataframe from 'create_epoch' function
    channel   : channel column to be analyzed
    hist_min  : minimum of histogram bin range
    hist_max  : maximum of histogram bin range
    num_bins  : number of bins you want
    """
    sweep_arrays = []
    epoch_arrays = []
    df_list = []
    sweeps = epoch_df.index.levels[0].values
    epochs = epoch_df.index.levels[1].values
    
    for sweep in sweeps:
        for epoch in epochs:
            data = epoch_df.ix[sweep][channel].xs(epoch)
            epoch_hist, bins = np.histogram(data, bins=num_bins, range=(hist_min,hist_max))
            arrays = [[sweep]*len(epoch_hist),[epoch]*len(epoch_hist),np.arange(len(bins))]
            index = pd.MultiIndex.from_arrays(arrays, names=['sweep','epoch',None])
            data_list = list(zip(bins, epoch_hist))
            df = pd.DataFrame(data_list, columns=['bin',channel])
            df.set_index(index, inplace=True)
            df_list.append(df)
    return pd.concat(df_list)


def epoch_kde(epoch_df, channel, range_min, range_max, samples=1000):
    """
    Creates a bunch of 1D KDEs of the epochs created from
    ea.create_epoch function.
    
    Input parameters
    epoch_df  : dataframe from 'create_epoch' function
    channel   : channel column to be analyzed
    range_min : minimum of KDE range
    range_max : maximum of KDE range
    samples   : number of KDE samples
    """
    
    df_list = []
    samples = samples
    x = np.linspace(range_min, range_max, samples)
    sweeps = epoch_df.index.levels[0].values
    epochs = epoch_df.index.levels[1].values
    
    for sweep in sweeps:
        for epoch in epochs:
            data = epoch_df.ix[sweep][channel].xs(epoch)
            kde = sp.stats.gaussian_kde(data)
            kde_data = kde(x)
            arrays = [[sweep]*len(x),[epoch]*len(x),np.arange(len(x))]
            index = pd.MultiIndex.from_arrays(arrays, names=['sweep','epoch',None])
            data_list = list(zip(x,kde_data))
            df = pd.DataFrame(data_list, columns=['x',channel])
            df.set_index(index, inplace=True)
            df_list.append(df)
    return pd.concat(df_list)


def epoch_pgram(epoch_df, channel, fs=10e3):
    """
    Run periodogram on each epoch

    Input parameters
    epoch_df  : dataframe from 'create_epoch' function
    channel   : channel column to be analyzed
    fs        : sampling frequency
    """
    
    df_list = []
    fs = fs
    sweeps = epoch_df.index.levels[0].values
    epochs = epoch_df.index.levels[1].values
    
    for sweep in sweeps:
        for epoch in epochs:
            data = epoch_df.ix[sweep][channel].xs(epoch)
            pgram_f, pgram_den = periodogram(data, fs)
            arrays = [[sweep]*len(pgram_f),[epoch]*len(pgram_f),np.arange(len(pgram_f))]
            index = pd.MultiIndex.from_arrays(arrays, names=['sweep','epoch',None])
            data_list = list(zip(pgram_f,pgram_den))
            df = pd.DataFrame(data_list, columns=['frequency',channel])
            df.set_index(index, inplace=True)
            df_list.append(df)
    return pd.concat(df_list)


def simpleaxis(ax):
    """
    (note: stolen from somewhere else, but forgot where)

    Removes the top and right axis lines and tick marks
    on standard pyplot figures. Does not work with 
    GridSpec objects (need to figure that out.)

    Input
    ax        : matplotlib.pyplot axis

    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def simpleaxes(ax):
    """
    THIS WILL BE REMOVED IN FUTURE RELEASES (PROBABLY...)

    Like 'simpleaxis,'' but for multiple subplots
    """
    for i, data in enumerate(ax):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].get_xaxis().tick_bottom()
        ax[i].get_yaxis().tick_left()