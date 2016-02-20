""" Functions to analyze pacemaking activity data """

import numpy as np
from . import utilities as util


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    ----------------------------------------------------------------------------
    Please note, this function is the work of Marcos Duarte.

    Citation:
    Duarte, M. (2015) Notes on Scientific Computing for Biomechanics and
    Motor Control. GitHub repository, https://github.com/demotu/BMC.
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) &
                           (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) &
                           (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1,
                                                    indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def baseline_pacemaking(df, n=200):
    """Baseline a pacemaking (cell attached) trace by subtracting the running
    average of the trace from the trace

    Parameters
    ----------
    df: data as pandas dataframe
        should contain time and primary columns
    n:  positive scalar, default = 200
        number of points to average for running average

    Return
    ------
    df: modified dataframe where primary column has been baselined

    Notes
    -----
    Simple_smoothing sets n-1 points to nan, which this function then sets to 0
    before subtracting from the data column. What this means is that those n-1
    values are not baselined. At the sampling frequencies normally used this
    should not be a major concern, though.
    """
    smoothed = util.simple_smoothing(df.primary.values, n)
    df.primary -= np.nan_to_num(smoothed)

    return df


def calc_freq(df, mph, valley=False, hz=True,
              ret_indices=False, ret_times=False):
    """Calculate instantaneous frequency of events exceeding a specific height

    Parameters
    ----------
    df: data as pandas dataframe
        should contain time and Primary columns
    mph: number (pA or mV)
        designates minimum height of event (i.e. threshold of event)
    valley : boolean, default = False
        if True, detect valleys (local minima) instead of peaks
    hz: boolean, default = True
        return frequency in Hz; if False, will return as ISI (seconds)
    ret_indices: boolean, default = False
        return the indices for the detected events
    ret_times: boolean, default = False
        return the times for the detected events

    Return
    ------
    if ret_indices == False and ret_times == False, return just
    frequencies (array)

    otherwise, will return a list (ret_vals) where ret_vals[0] is the
    frequency array. indices and times are returned ordered in that
    order in list if both are desired (i.e. ret_vals[1] and ret_vals[2])
    """
    ret_vals = []
    if valley:
        indices = detect_peaks(df['primary'].values, mph=abs(mph),
                               valley=valley, mpd=100)
    else:
        indices = detect_peaks(df['primary'].values, mph=mph, mpd=100)
    times = df['time'].ix[indices].values
    times_dif = times[1:] - times[:-1]

    if hz:
        ret_vals.append(1/times_dif)
    else:
        ret_vals.append(times_dif)

    if ret_indices:
        ret_vals.append(indices)
    if ret_times:
        ret_vals.append(times[1:])

    return ret_vals[0] if len(ret_vals) == 1 else ret_vals
