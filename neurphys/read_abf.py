"""
Functions to import and manipulate Axon Binary Files.
"""

from neo import io
import pandas as pd

def _all_ints(ii):
    """
    list or tuple

    """

    return all(isinstance(i, int) for i in ii)


def _all_strs(ii):
    """ list or tuple """

    return all(isinstance(i, str) for i in ii)


def read_abf(filepath):
    """
    Imports ABF file using neo io AxonIO, breaks it down by blocks
    which are then processed into a multidimensional pandas dataframe
    where each block corresponds to a sweep and columns represent time
    and each recorded channel.

    Parameters
    ----------
    filename: str
        Full filepath WITH '.abf' extension.

    Return
    ------
    df: DataFrame
        Pandas DataFrame broken down by sweep.

    References
    ----------
    [1] https://neo.readthedocs.org/en/latest/index.html
    """

    r = io.AxonIO(filename=filepath)
    bl = r.read_block(lazy=False, cascade=True)
    num_channels = len(bl.segments[0].analogsignals)
    channels = ['primary']
    df_list = []
    signals = []
    sweep_list = []

    for seg_num, seg in enumerate(bl.segments):
        channels = ['primary']+['channel_{0}'.format(str(i+1)) for i in range(num_channels-1)]
        signals = []
        for i in range(num_channels):
            signals.append(bl.segments[seg_num].analogsignals[i])
        data_dict = dict(zip(channels, signals))
        time = seg.analogsignals[0].times - seg.analogsignals[0].times[0]
        data_dict['time'] = time
        df = pd.DataFrame(data_dict)
        df_list.append(df)
        sweep_list.append('sweep' + str(seg_num + 1).zfill(3))
    df = pd.concat(df_list, keys=sweep_list, names=['sweep'])

    return df


def keep_sweeps(df, sweep_list):
    """
    Keeps specified sweeps from your DataFrame.

    Parameters
    ----------
    df: Pandas DataFrame
        Dataframe created using one of the functions from Neurphys.
    sweep_list: 1D array_like of ints or properly formatted strings
        List containing numbers of the sweeps you'd like dropped from the
        DataFrame. Example: [1,4,6] or ['sweep001', 'sweep004', 'sweep006']

    Return
    ------
    keep_df: Pandas Dataframe
        Dataframe containing only the sweeps you want to keep.

    Notes
    -----
    Some type checks are made, but not enough to cover the plethora of
    potential inputs, so read the docs if you're having trouble.
    """

    if _all_ints(sweep_list):
        keep_sweeps = [('sweep'+str(i).zfill(3)) for i in sweep_list]
    elif _all_strs(sweep_list):
        keep_sweeps = sweep_list
    else:
        raise TypeError(
        'List should either be appropriate sweep names or integers')
    keep_df = df.loc[keep_sweeps]

    return keep_df


def drop_sweeps(df, sweep_list):
    """
    Removes specified sweeps from your DataFrame.

    Parameters
    ----------
    df: Pandas DataFrame
        Dataframe created using one of the functions from Neurphys.
    sweep_list: 1D array_like of ints
        List containing numbers of the sweeps you'd like dropped from the
        DataFrame.

    Return
    ------
    drop_df: Pandas Dataframe
        Dataframe containing only the sweeps you want to keep.

    Notes
    -----
    Making the grand assumption that the df.index.level[0]=='sweeps'.

    TODO
    ----
    As always, I need to build exceptions into this function and just need to
    make it generally more robust.
    - make possibility for 'sweep_list' to be either pure numerical list, or
    actually contain list of sweep names (ex. ['sweep002','sweep016'])
    """

    all_sweeps = df.index.levels[0].values
    drop_sweeps = [('sweep'+str(i).zfill(3)) for i in sweep_list]
    # keep_sweeps = list(set(all_sweeps).difference(drop_sweeps))
    keep_sweeps = list(set(all_sweeps) ^ set(drop_sweeps))
    drop_df = df.loc[pd.IndexSlice[keep_sweeps]]

    return drop_df
