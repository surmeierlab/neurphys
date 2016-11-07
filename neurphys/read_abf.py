"""
Functions to import and manipulate Axon Binary Files.
"""

from neo import io
import pandas as pd


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
    signals = []
    df_list = []
    sweep_list = []

    for seg_num, seg in enumerate(bl.segments):
        for i in range(num_channels):
            signals.append(bl.segments[seg_num].analogsignals[i])
            if i >= 1:
                channels.append('channel_'+str(i))
        data_dict = dict(zip(channels, signals))
        time = seg.analogsignals[0].times - seg.analogsignals[0].times[0]
        data_dict['time'] = time
        df = pd.DataFrame(data_dict)
        df_list.append(df)
        sweep_list.append('sweep'+str(seg_num+1).zfill(3))
    df = pd.concat(df_list, keys=sweep_list, names=['sweep'])
    return df


def keep_sweeps(df, sweep_list):
    """
    Keeps specified sweeps from your DataFrame.

    Parameters
    ----------
    df: Pandas DataFrame
        Dataframe created using one of the functions from Neurphys.
    sweep_list: 1D array_like of ints
        List containing numbers of the sweeps you'd like dropped from the
        DataFrame.

    Return
    ------
    keep_df: Pandas Dataframe
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

    keep_sweeps = [('sweep'+str(i).zfill(3)) for i in sweep_list]
    keep_df = df.loc[pd.IndexSlice[keep_sweeps]]
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
