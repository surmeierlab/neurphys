""" Functions for analyzing membrane properties of a cell """

from scipy.integrate import trapz
from . import utilities as util


def calc_mem_prop(df, bsl_start, bsl_end, pulse_start, pulse_dur, pulse_amp):
    """Fit capacitive transient to calculate membrane access resistances (ra),
    membrane resistance (rm), membrane capacitance (cm), and membrane time
    constant (tau)

    Input Parameters
    -----------------
    df: data as pandas dataframe
        should contain Time and Primary columns
    bsl_start: positive number (time, seconds)
        designates beginning of epoch to use to baseline data
    bsl_end: positive number (time, seconds)
        designates end of epoch (time) to use to baseline data
    pulse_start: positive number (time, seconds)
        designates beginning voltage step associated with capacitive transient
    pulse_dur: positive number (time, seconds)
        duration of the voltage step associated with capacitive transient
    pulse_amp: positive or negative number (mV)
        amplitude of the voltage step (with appropriate sign, - or +)

    Return
    ------
    ra: access resistance (MOhm)
    rm: membrane resistance (MOhm)
    cm: membrane capacitance (pF)
    tau: membrane time constant (ms)

    References
    ----------
    For associated equations, see:
    pClamp 10: Data Acquisition and Analysis for Comprehensive
    Electrophysiology - User Guide, pages 163-166.
    """
    # have to make copy of df to not modify original df with calculation
    data = df.copy()

    # conversions - pulse_amp is in mVs, data.Primary is in pAs
    pulse_amp *= 1e-3
    data.Primary *= 1e-12

    # baseline recording data
    data = util.baseline(data, bsl_start, bsl_end)

    # i_baseline is defined as average current over baseline region
    i_baseline = data.Primary[(data.Time >= bsl_start) &
                              (data.Time <= bsl_end)].mean()

    # i_ss is the stead-state current during the pulse, taken between 70% and
    # 90% of pulse duration
    i_ss = data.Primary[(data.Time >= pulse_start + pulse_dur*0.7) &
                        (data.Time <= pulse_start + pulse_dur*0.9)].mean()

    # calculate delta_i -- i.e. difference between baseline current amplitude
    # and steady-state current amplitude
    delta_i = (i_ss-i_baseline)

    # remove delta_i; this part of the capacitance charge is calculated
    # later as q2
    data.Primary -= delta_i

    pulse_end = pulse_start + pulse_dur
    if pulse_amp > 0:
        peak_df = util.find_peak(data, pulse_start, pulse_end, 'max')

    elif pulse_amp < 0:
        peak_df = util.find_peak(data, pulse_start, pulse_end, 'min')

    peak = peak_df['Peak Amp'].values[0]
    peak_time = peak_df['Peak Time'].values[0]
    tau, x_vals, y_vals, fit_vals = util.calc_decay(data, peak, peak_time, True)

    # take integral of curve to get q1
    q1 = trapz(fit_vals, x_vals)

    # q2 is correction for charge from i_baseline to i_ss
    q2 = tau * delta_i

    # total charge
    qt = q1 + q2

    # resistance calculations
    ra = (tau*pulse_amp)/qt
    rt = (pulse_amp)/delta_i
    rm = rt - ra

    # capacitance calculation
    cm = (qt * rt) / (pulse_amp * rm)

    return ra*1e-6, rm*1e-6, cm*1e12, tau*1e3
x
