""" Useful functions for performing ephys data analysis """

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def baseline(df, start_time, end_time):
    """Subtracts from entire data column average of subset of data column
    defined by start and end times.

    Parameters
    -----------
    df: data as pandas dataframe
        should contain time and primary columns
    start_time: positive number (seconds)
        designates beginning of the region over which to average
    end_time: positive number (seconds)
        designates end of region over which to average

    Return
    ------
    df: dataframe with modified primary column
    """
    avg = df.primary[(df.time >= start_time) & (df.time <= end_time)].mean()
    df.primary -= avg

    return df


def find_peak(df, start_time, end_time, sign="min"):
    """Returns min (or max) of data subset as a dataframe

    Parameters
    -----------
    df: data as pandas dataframe
        should contain time and primary columns
    start_time: positive number (seconds)
        designates beginning of the epoch in which the event occurs
    end_time: positive number (seconds)
        designates end of the epoch in which the event occurs
    sign: string (either 'min' or 'max')
        indicates direction of event (min = neg going, max = pos going)

    Return
    -------
    peak_df: dataframe of Peak Amp and Peak time
    """
    df_sub = df[(df.time >= start_time) & (df.time <= end_time)]
    if sign == "min":
        peak = df_sub.primary.min()
    elif sign == "max":
        peak = df_sub.primary.max()

    peak_df = df_sub[df_sub.primary == peak][['time', 'primary']]
    peak_df.columns = ['Peak time', 'Peak Amp']

    return peak_df.head(1)


def calc_decay(df, peak, peak_time, return_plot_vals=False):
    """Performs biexponential fit of event, returns a weighted tao value

    Parameters
    -----------
    df: data as pandas dataframe
        should contain time and primary columns
    peak: scalar (pA or mV)
        amplitude of event
    peak_time: positive scalar (seconds)
        time at which the peak occurs
    return_plot_vals: boolean, default = False
        return x, y, and fit_y values for fit associated with tau calculation

    Notes
    -----
    It is assumed that the primary column in the passed df have been baselined

    Return
    ------
    tau: weighted tau (unit = ms)
    **if return_plot_values == True:
        also return subset of 1. x values (time), 2. subet of
        y values (primary) and 3. the y-data for the fit. Useful for plotting
        the fit overlayed with the raw data.
    """
    peak_sub = df[df.time >= peak_time]

    if peak < 0:
        index1 = peak_sub[peak_sub.primary >= peak * 0.90].index[0]
        index2 = peak_sub[peak_sub.primary >= peak * 0.05].index[0]
        fit_sub = peak_sub.ix[index1:index2]
        guess = np.array([-1, 1, -1, 1, 0])
    else:
        index1 = peak_sub[peak_sub.primary <= peak * 0.90].index[0]
        index2 = peak_sub[peak_sub.primary <= peak * 0.05].index[0]
        fit_sub = peak_sub.ix[index1:index2]
        guess = np.array([1, 1, 1, 1, 0])
    
    x_zeroed = fit_sub.time - fit_sub.time.values[0]
    
    def exp_decay(x, a, b, c, d, e):
        return a*np.exp(-x/b) + c*np.exp(-x/d) + e

    popt, pcov = curve_fit(exp_decay, x_zeroed*1e3, fit_sub.primary*1e12, guess)

    x_full_zeroed = peak_sub.time - peak_sub.time.values[0]
    y_curve = exp_decay(x_full_zeroed*1e3, *popt) / 1e12

    amp1 = popt[0]
    tau1 = popt[1]
    amp2 = popt[2]
    tau2 = popt[3]

    tau = ((tau1*amp1)+(tau2*amp2))/(amp1+amp2) * 1e-3

    if return_plot_vals:
        return tau, x_full_zeroed, peak_sub.primary, y_curve
    else:
        return tau


def simple_smoothing(data, n):
    """Calculates running average of n data points

    Parameters
    ----------
    data: 1D array
    n: positive scalar

    Notes
    -----
    to return array of same length as data array, n-1 nan values
    are placed at the start of the return array

    Return:
    1D array of same length as input array (data)
    """
    if np.isnan(np.sum(data)):
        starting_nans = data[np.isnan(data)]
        data_no_nan = data[len(starting_nans):]

        smoothed = simple_smoothing(data_no_nan, n)

        return np.append(starting_nans, smoothed)

    else:
        cum_sum = np.cumsum(data, dtype=float)
        cum_sum[n:] = cum_sum[n:] - cum_sum[:-n]

        nan_array = np.full(n-1, np.nan)
        smoothed = cum_sum[n - 1:] / n

        return np.append(nan_array, smoothed)


def _mock_df(rows = 20, num_channels = 2):
    """
    Make a mock DataFrame that mimics neurphys.read_abf
    dataframe for testing purposes. Assuming at 10kHz sampling rate.

    Parameters
    ----------
    rows: int (default: 20)
    num_channels: int (default: 2)

    Return
    ------
    d: pd.DataFrame
        Pandas Dataframe

    Note
    ----
    Could do assertion checks, but nope. Not gonna do it.
    """

    d = {'channel_{}'.format(channel): np.random.randn(rows) for channel in range(num_channels)}
    d['primary'] = np.random.randn(rows)
    d['time'] = np.arange(0,(rows*0.0001),0.0001)

    return pd.DataFrame(d)


def mock_multidf(rows = 20, num_channels = 2, num_sweeps = 10):
    """
    Make a mock DataFrame that mimics neurphys.read_abf
    dataframe for testing purposes. Assuming at 10kHz sampling rate.

    Parameters
    ----------
    rows: int (default: 20)
    num_channels: int (default: 2)
    sweeps: int (default: 10)

    Note
    ----
    Could do assertion checks, but nope. Not gonna do it.
    """

    df_dict = {}
    sweep_names = ['sweep{}'.format(str(sweep+1).zfill(3)) for sweep in range(num_sweeps)]

    for sweep in sweep_names:
        df_dict[sweep] = _mock_df(rows=rows, num_channels=num_channels)

    return pd.concat(df_dict, names=['sweep'])
