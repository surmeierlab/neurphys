""" Useful functions for performing ephys data analysis """

import numpy as np
from scipy.optimize import curve_fit


def baseline(df, start_time, end_time):
    """Subtracts from entire data column average of subset of data column
    defined by start and end times.
    """
    
    avg = df.Primary[(df.Time >= start_time) & (df.Time <= end_time)].mean()
    df.Primary -= avg

    return df

def find_peak(df, start_time, end_time, sign="min"):
    """Returns min (or max) of data subset as a dataframe

    df: data as pandas dataframe, should contain Time and Primary columns
    start_time: designates beginning of the epoch in which the event occurs
    end_time: designates end of the epoch in which the event occurs 
    sign: takes two values - 'min' or 'max' - depending on direction of event

    """
    df_sub = df[(df.Time >= start_time) & (df.Time <= end_time)]
    if sign == "min":
        peak = df_sub.Primary.min()
    elif sign == "max":
        peak = df_sub.Primary.max()

    peak_df = df[df.Primary == peak][['Time', 'Primary']]
    peak_df.columns = ['Peak Time', 'Peak Amp']
    
    return peak_df.tail(1)

def calc_decay(df, peak, peak_time, return_plot_vals=False):
    """Performs biexponential fit of event, returns a weighted tao value

    df: data as pandas dataframe, should contain Time and Primary columns. 
    *Note that it is assumed that the Primary column in the passed df is assumed
    to have alredady been baselined
    peak: amplitude of event
    peak_time: time at which the peak occurs
    return_plot_values: if True, will return 1. weighted tao, 2. subset of x
    values (Time), 3. subet of y values (Primary) and 4. the y-data for the fit.
    Useful for plotting the fit overlayed with the raw data.
    """
    peak_sub = df[df.Time >= peak_time]

    if peak < 0:
        index1 = peak_sub[peak_sub.Primary >= peak * 0.90].index[0]
        index2 = peak_sub[peak_sub.Primary >= peak * 0.05].index[0]
        fit_sub = peak_sub.ix[index1:index2]
        guess = np.array([-1, 1e3, -1, 1e3])

    else:
        index1 = peak_sub[peak_sub.Primary <= peak * 0.90].index[0]
        index2 = peak_sub[peak_sub.Primary <= peak * 0.05].index[0]
        fit_sub = peak_sub.ix[index1:index2]
        guess = np.array([1, 1e3, 1, 1e3])

    x_zeroed = fit_sub.Time - fit_sub.Time.values[0]

    def exp_decay(x, a, b, c, d):
        return a*np.exp(-b*x) + c*np.exp(-d*x)

    popt, pcov = curve_fit(exp_decay, x_zeroed, fit_sub.Primary, guess)

    x_full_zeroed = peak_sub.Time - peak_sub.Time.values[0]
    y_curve = exp_decay(x_full_zeroed, *popt)

    amp1 = popt[0]
    tau1 = 1/popt[1]
    amp2 = popt[2]
    tau2 = 1/popt[3]

    tau = ((tau1*amp1)+(tau2*amp2))/(amp1+amp2)

    if return_plot_vals:
        return tau, x_full_zeroed, peak_sub.Primary, y_curve
    else:
        return tau


def simple_smoothing(data, n):
    """Calculates running average of n data points, returning array of same
    length as input array but with first n-1 data points set to nan
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
