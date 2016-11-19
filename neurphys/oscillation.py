"""
Functions for analyzing oscillatory activity.
"""

import numpy as np
from scipy.signal import periodogram
from scipy.stats import gaussian_kde
import pandas as pd


def _create_epoch(df, window, step):
    """
    Creates a generator that blocks a Pandas DataFrame based on your
    input parameters

    Parameters
    ----------
    df: DataFrame
        Pandas Dataframe from 'read_abf/pv' function.
    window: int
        Epoch size based on array index.
    step: int
        Start-to-start number of rows between captured windows (may
        overlap with other windows).

    Yields
    -------
    epoch_df: DataFrame
        Multiindexed Pandas DataFrame with column names unchanged and
        an added index level named 'epoch.'

    Notes
    -----
    To keep the funtion as simple as possible, the number of epochs
    has been limited to 999, but if you really need more then feel
    free to change the source code.

    Based on your specified window and step size, your new DataFrame
    may be truncated.
    """

    window, step = int(window), int(step)
    # if multiple sweeps, assuming all sweeps are exactly the same length
    num_rows = df.index.levshape[1]
    num_epochs = int(1 + (num_rows - window) / step)
    if num_epochs > 1000:
        raise ValueError(
            'Too many epochs. Change parameters to create <1000 epochs')
    sweeps = df.index.levels[0].values

    for sweep in sweeps:
        # first index out a single sweep to prevent runover
        sweep_df = df.ix[sweep]
        for epoch in range(num_epochs):  # faster than a while loop
            # have to add 0 to beginning so the first multiplication
            # has a real number result
            data = sweep_df[(0 + step * epoch):(window + step * epoch)]
            yield data


def _epoch_data(df, window, step):
    """
    Returns a few useful parameters about epoch dataframes to use for
    downstream functions.

    Parameters
    ----------
    df: DataFrame
        Pandas Dataframe from 'read_abf/pv' function.
    window: int
        Epoch size based on array index.
    step: int
        Start-to-start number of rows between captured windows (may
        overlap with other windows).

    Returns
    -------
    sweep_names: list
        List of sweep names from the input DataFrame.
    epoch_names:
        List of epoch names made by the _create_epoch function.

    """

    num_rows = df.index.levshape[1]
    num_epochs = int(1 + (num_rows - window) / step)
    sweep_names = df.index.levels[0].values
    epoch_names = ['epoch{}'.format(str(i+1).zfill(3))
                   for i in range(num_epochs)]
    return sweep_names, epoch_names


def epoch_hist(df, window, step, channel, hist_min, hist_max, num_bins):
    """
    Create a 1D histogram for each epoch based on input parameters.

    Parameters
    ----------
    df: DataFrame
        Pandas Dataframe from 'read_abf/pv' function.
    window: int
        Epoch size based on array index.
    step: int
        Start-to-start number of rows between captured windows (may
        overlap with other windows).
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
    df: DataFrame
        Multiindexed Pandas DataFrame with index levels unchanged, but
        with an added column specifiying the bin for each value.

    Notes
    -----
    'bins' column contains the 'leftmost' bin edge. Also note, that
    bins are truncated from original numpy function of (len(hist + 1)).
    Check numpy docs if confused.
    """

    # set up basic containers and inputs
    hist_arrays = []
    bin_arrays = []
    epochs = _create_epoch(df, window, step)
    sweep_names, epoch_names = _epoch_data(df, window, step)

    # set up indicies for returned df (just easier to remake them here)
    idx = np.arange(num_bins)
    arrays = [sweep_names, epoch_names, idx]
    index = pd.MultiIndex.from_product(arrays, names=['sweep', 'epoch', None])

    for epoch in epochs:
        hist, bins = np.histogram(epoch[channel],
                                  bins=num_bins, range=(hist_min, hist_max))
        hist_arrays.append(hist)
        bin_arrays.append(bins[:-1])

    # stitch the arrays together
    hist_concat = np.concatenate(hist_arrays, axis=0)
    bin_concat = np.concatenate(bin_arrays, axis=0)
    data = list(zip(bin_concat, hist_concat))

    # turn them into a dataframe with the correct column labels
    df = pd.DataFrame(data, columns=['bin', channel])

    # add the correct multiindex labeling
    df.set_index(index, inplace=True)

    return df


def epoch_kde(df, window, step, channel, range_min, range_max,
              resolution=None):
    """
    Returns a 1D kernel density estimation with automatic bandwidth
    detection for each of the epochs created from the 'create_epoch'
    function.

    Parameters
    ----------
    df: DataFrame
        Pandas Dataframe from 'read_abf/pv' function.
    window: int
        Epoch size based on array index.
    step: int
        Start-to-start number of rows between captured windows (may
        overlap with other windows).
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
    df: Dataframe
        Multiindexed Pandas DataFrame with index levels unchanged, but
        with an added column specifiying the x value for each
        corresponding density value (similar to the 'bin' in the
        histogram function, but not exactly the same)

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/
    scipy.stats.gaussian_kde.html
    """

    # set up basic containers and inputs
    kde_arrays = []
    x_arrays = []
    if resolution is None:
        resolution = abs(range_min - range_max) * 5
    else:
        resolution = resolution

    epochs = _create_epoch(df, window, step)
    sweep_names, epoch_names = _epoch_data(df, window, step)

    # set up indicies for returned df (just easier to remake them here)
    x = np.linspace(range_min, range_max, resolution)
    idx = np.arange(resolution)
    arrays = [sweep_names, epoch_names, idx]
    index = pd.MultiIndex.from_product(arrays, names=['sweep', 'epoch', None])

    for epoch in epochs:
        kde = gaussian_kde(epoch[channel])
        kde_data = kde(x)
        kde_arrays.append(kde_data)
        x_arrays.append(x)

    # stitch the arrays together
    kde_concat = np.concatenate(kde_arrays, axis=0)
    x_concat = np.concatenate(x_arrays, axis=0)
    data = list(zip(x_concat, kde_concat))

    # turn them into a dataframe with the correct column labels
    df = pd.DataFrame(data, columns=['x', channel])

    # add the correct multiindex labeling
    df.set_index(index, inplace=True)
    return df


def epoch_pgram(df, window, step, channel, fs=10e3):
    """
    Returns a periodogram for each of the epochs created from the
    'create_epoch' function.

    Parameters
    ----------
    df: DataFrame
        Pandas Dataframe from 'read_abf/pv' function.
    window: int
        Epoch size based on array index.
    step: int
        Start-to-start number of rows between captured windows (may
        overlap with other windows).
    channel: str
        Channel column to be analyzed.
    fs: int (default: 10000)
        Sampling frequency (Hz).

    Returns
    -------
    df: Dataframe
        Multiindexed Pandas DataFrame with the estimated power spectral
        density (V^2/Hz) and frequency as the new column names. All
        indexing from the input DataFrame remain unchanged.

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/
    scipy.signal.periodogram.html
    """

    # set up basic containers and inputs
    f_arrays = []
    den_arrays = []
    fs = int(fs)

    epochs = _create_epoch(df, window, step)
    sweep_names, epoch_names = _epoch_data(df, window, step)

    # set up indicies for returned dataframe (just easier to remake them here)
    idx = np.arange((window/2)+1)
    arrays = [sweep_names, epoch_names, idx]
    index = pd.MultiIndex.from_product(arrays, names=['sweep', 'epoch', None])

    for epoch in epochs:
        f, den = periodogram(epoch[channel], fs)
        f_arrays.append(f)
        den_arrays.append(den)

    # stitch the arrays together
    f_concat = np.concatenate(f_arrays, axis=0)
    den_concat = np.concatenate(den_arrays, axis=0)
    data = list(zip(f_concat, den_concat))

    # turn them into a dataframe with the correct column labels
    df = pd.DataFrame(data, columns=['frequency', channel])

    # add the correct multiindex labeling
    df.set_index(index, inplace=True)
    return df


def nu_spectrogram(df, window, step, channel, fs, f_trim=(0,100)):
    """
    Parameters
    ----------
    df: DataFrame
        Pandas Dataframe from 'read_abf/pv' function.
    window: int
        Epoch size based on array index.
    step: int
        Start-to-start number of rows between captured windows (may
        overlap with other windows).
    channel: str
        Channel column to be analyzed.
    fs: int (default: 10000)
        Sampling frequency (Hz).
    f_trim: tuple
        Range of returned frequency bands.

    Returns
    -------
    df :
        Spectrogram of DataFrame column labeled with frequencies added as row
        indicies and columns as segment times (left aligned).
    """

    noverlap = nperseg - step
    f_trim = f_trim
    # make sure window and step are integers
    window, step = int(window), int(step)
    df_list = []  # change to a dict once Py3.6 becomes common?
    sweeps = df.index.levels[0].values

    for sweep in sweeps:
        # compute the spectrogram. f=sample frequencies, t=segment times
        f, t, Sxx = signal.spectrogram(df[channel].xs(sweep),
                                       fs, nperseg=window, noverlap=noverlap)
        t -= t[0]  # left align the spectrogram time values
        df_list.append(
        pd.DataFrame(Sxx,index=f,columns=t).loc[f_trim[0]:f_trim[1]])

    df = pd.concat(df_list, keys=sweeps, names=['sweep'])
    return df
