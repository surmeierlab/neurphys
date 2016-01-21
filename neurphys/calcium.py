""" Module for analyzing 2PLSM calcium imaging data """


def calc_ca_conc(df, profile_num, f0_start, f0_end, background,
                 kd, rf, rf_real):
    """Calculates estimated calcium concentration from profile fluorescent
    values and systema and dye properties.

    Parameters
    -----------
    df: data as pandas dataframe
        Profiles are named with the convention Prof 1, Prof 2, etc. with
        corresponding Prof 1 Time, Prof 2 Time, etc. columns
    profile_num: positive number
        Profile number (1, 2, etc) that identifies which profile contains the
        fluorescent values to be used to calculate calcium concentration
    f0_start: positive number (seconds)
        designates beginning of the region over which to average to determine f0
    f0_end: positive number (seconds)
        designates end of the region over which to average to determine f0
    background: positive number (AU)
        PMT background fluorescent value
    kd: positive number (nM)
        kd for specific dye being used in experiment
    rf: positive number (no units)
        theoretical rf for specific dye being used in experiment
    rf_real: positive number (no units)
        experimentally determined rf for specific dye being used

    Return
    ------
    1D array of calcium concentration (nM) values
    """

    prof = "Prof " + str(profile_num)
    prof_time = prof + " Time"

    ca_df = df[[prof, prof_time]].copy()
    ca_df[prof] -= background

    f0 = ca_df[prof][(ca_df[prof_time] >= f0_start) &
                     (ca_df[prof_time] <= f0_end)].mean()
    fmax = f0 * (rf / rf_real)
    ca_df[prof] = kd * ((1-ca_df[prof] / fmax) / (ca_df[prof] / fmax - (1/rf)))

    return ca_df[prof].values
