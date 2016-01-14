""" Functions to analyze synaptic events """

import pandas as pd
from . import utilities as util


def analyze_current(df, bsl_start, bsl_end, start_time, end_time, sign="min",
                    calc_tau=False, tau_plot=False):
    """Calculate peak amplitude and (optionall) decay of a synpaptic current

    Input Parameters
    -----------------
    df: data as pandas dataframe
        should contain Time and Primary columns
    bsl_start: positive number (seconds)
        designates beginning of epoch to use to baseline data
    bsl_end: positive number (seconds)
        designates end of epoch (time) to use to baseline data
    start_time: positive number (seconds)
        designates beginning of the epoch in which the event occurs
    end_time: positive number (seconds)
        designates end of the epoch in which the event occurs
    sign: string (either 'min' or 'max')
        indicates direction of event (min = neg going, max = pos going)
    calc_tau: boolean, default = False
        calculate tau of event
    tau_plot: boolean, default = False
        return x, y, and fit_y values for fit associated with tau calculation

    Return
    ------
    peak_df: dataframe of Peak Amp and Peak Time
    **if calc_tau == True:
         also return weighted tau (unit = ms)
    **if tau_plot == True:
         also return subset of 1. x values (Time), 2. subet of
         y values (Primary) and 3. the y-data for the fit. Useful for plotting
         the fit overlayed with the raw data.
    ***Note that if tau_plot is True, weighted tau will be returned even if
    calc_tau hasn't been set to True
    """
    df_bsl = util.baseline(df, bsl_start, bsl_end)
    peak_df = util.find_peak(df_bsl, start_time, end_time)

    if tau_plot:
        peak = peak_df['Peak Amp'].values[0]
        peak_time = peak_df['Peak Time'].values[0]
        tau, x_vals, y_vals, fit_vals = util.calc_decay(df_bsl, peak,
                                                        peak_time, tau_plot)
        return peak_df, tau*1e3, x_vals, y_vals, fit_vals

    elif calc_tau:
        peak = peak_df['Peak Amp'].values[0]
        peak_time = peak_df['Peak Time'].values[0]
        tau = util.calc_decay(df_bsl, peak, peak_time)
        return peak_df, tau*1e3

    else:
        return peak_df


def calc_ppr(df, bsl_start, bsl_end, start_time, end_time, stim_interval,
             sign="min"):
    """Calculate paired-pulse ratio from current peaks

    Input Parameters
    -----------------
    df: data as pandas dataframe
        should contain Time and Primary columns
    bsl_start: positive number (seconds)
        designates beginning of epoch to use to baseline data
    bsl_end: positive number (seconds)
        designates end of epoch (time) to use to baseline data
    start_time: positive number (seconds)
        designates beginning of the epoch in which the event occurs
    end_time: positive number (seconds)
        designates end of the epoch in which the event occurs
    stim_interval: positive number (seconds)
        time between first stimulus and second stimulus
    sign: string (either 'min' or 'max')
        indicates direction of event (min = neg going, max = pos going)

    Return
    ------
    ppr_df: dataframe containing Peak 1 ampltiude, Peak 2 amplitude, and PPR
    """
    first_end = start_time + stim_interval

    df_bsl = util.baseline(df, bsl_start, bsl_end)

    # nudge first end to the left so that the two regions don't overlap
    peak1 = util.find_peak(df_bsl, start_time, first_end-0.0001,
                           sign=sign)["Peak"].values
    peak2 = util.find_peak(df_bsl, first_end, end_time,
                           sign=sign)["Peak"].values

    ppr = peak2 / peak1
    ppr_df = pd.DataFrame({"Peak 1": peak1, "Peak 2": peak2, "PPR": ppr})

    return ppr_df
