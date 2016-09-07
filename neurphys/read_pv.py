
""" Module for importing PraireView5.0+ generated .csv files."""

import os
import pandas as pd
from lxml import etree
from glob import glob


def _get_ephys_vals(element):
    """Gets conversion values for primary and secondary
    Multiclamp channels"""

    ch_type = element.find('.//PatchclampChannel').text

    if ch_type == '0':
        unit = element.find('.//UnitName').text
        divisor = float(element.find('.//Divisor').text)

        return {'unit': unit, 'divisor': divisor}

    elif ch_type == '1':
        unit = element.find('.//UnitName').text
        divisor = float(element.find('.//Divisor').text)

        return {'unit': unit, 'divisor': divisor}


def parse_xml(filename):
    """Parses VoltageRecording .xml file to get experiment metadata"""

    tree = etree.parse(filename)
    # find all elements associated with enabled channels
    enabled_ch = tree.xpath('.//Enabled[text()="true"]')

    file_attr = {}
    ch_names = []
    for ch in enabled_ch:
        parent = ch.getparent()
        if parent.find('.//Type').text == 'Physical':
            clamp_device = parent.find('.//PatchclampDevice').text
            name = parent.find('.//Name').text
            
            if clamp_device is not None:
                ephys_vals = _get_ephys_vals(parent)
                file_attr[name] = ephys_vals

            ch_names.append(name.lower())

    file_attr['channels'] = ch_names
    # gets sampling rate
    file_attr['sampling'] = int((tree.find('.//Rate')).text)
    # gets recording time, converts to sec
    file_attr['duration'] = (int((tree.find('.//AcquisitionTime')).text))/1000

    # finds the voltage recording csv file name
    datafile = (tree.find('.//DataFile')).text
    # finds the linescan profile file name (if doesn't exist, will be None)
    ls_file = (tree.find('.//AssociatedLinescanProfileFile')).text

    # If ls_file is none this could mean that there is no linescan associated
    # with that voltage recording file or that the file passed to parse_vr is
    # actually a LineScan data file and therefore should be passed to ls_file.
    # In that scenario there is no voltage recording file, so vo_file is None
    if ls_file is None:
        if "LineScan" in datafile:
            ls_file = datafile
            vo_file = None
        elif "LineScan" not in datafile:
            vo_file = datafile
    else:
        vo_file = datafile

    file_attr['voltage recording file'] = vo_file
    file_attr['linescan file'] = ls_file

    return file_attr


def import_vr_csv(filename, col_names, primary_div=1, secondary_div=1):
    """
    Reads voltage recording .csv file into a pandas dataframe.
    Will convert primary and secondary channels to appropriate values if those
    channels are in the file.

    Returns a dataframe
    """

    df = pd.read_csv(filename, names=col_names, skiprows=1)
    df.time /= 1000

    if "primary" in df.columns:
        df.primary /= primary_div
    if "secondary" in df.columns:
        df.secondary /= secondary_div

    return df


def import_ls_csv(filename):
    """
    Reads linescan profile .csv file into pandas dataframe.
    Returns a dataframe
    """

    df = pd.read_csv(filename, skipinitialspace=True)
    df.rename(columns=lambda header: header.strip('(ms)'), inplace=True)
    # time columns occur as every other column, starting with column 0
    df.ix[:, ::2] /= 1000

    return df


def import_folder(folder):
    """Collapse entire data folder into multidimensional dataframe

    Parameters
    ----------
    folder: string
        Full path to data folder. Folder must contain, at a minimum
        a single VoltageRecording XML file and associated csv file

    Return
    ------
    output: dictionary
        keys: 'voltage recording': contains voltage recording dataframe (if any
               voltage recording data in folder)
              'linescan': contains linescane dataframe (if any line scan data
               data in folder)
              'file attributes': dictionary of contain file: file attribute
               key: value pairs (file attributes from parsing XML)
    """
    vr_xmls = sorted(glob(os.path.join(folder, '*_VoltageRecording_*.xml')))

    if any(vr_xmls):
        data_vr = []
        data_ls = []
        sweep_list = []
        file_attr = {}
        output = {}

        for i, file in enumerate(vr_xmls):
            sweep = 'Sweep' + str(i+1).zfill(4)
            sweep_list.append(sweep)
            file_vals = parse_xml(file)

            if file_vals['voltage recording file'] is not None:
                vr_filename = os.path.join(folder,
                                           (file_vals['voltage recording file']
                                            + '.csv'))
                col_names = ['time'] + file_vals['channels']
                if 'primary' in file_vals.keys():
                    primary_divisor = file_vals['primary']['divisor']
                else:
                    primary_divisor = 1
                if 'secondary' in file_vals.keys():
                    secondary_divisor = file_vals['secondary']['divisor']
                else:
                    secondary_divisor = 1

                df_vr = import_vr_csv(vr_filename, col_names, primary_divisor,
                                      secondary_divisor)

                data_vr.append(df_vr)

            if file_vals['linescan file'] is not None:
                ls_filename = os.path.join(folder,
                                           (file_vals['linescan file']))

                df_ls = import_ls_csv(ls_filename)

                data_ls.append(df_ls)

            file_attr['File'+str(i+1)] = file_vals

        if data_vr:
            output["voltage recording"] = pd.concat(data_vr, keys=sweep_list,
                                                    names=['Sweep', 'Index'])
        elif not data_vr:
            output["voltage recording"] = None
        if data_ls:
            output["linescan"] = pd.concat(data_ls, keys=sweep_list,
                                           names=['Sweep', 'Index'])
        elif not data_ls:
            output["linescan"] = None
        output["file attributes"] = file_attr

    else:
        output = {"voltage recording": None, "linescan": None,
                  "file attributes": None}

    return output
