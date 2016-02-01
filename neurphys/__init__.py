"""
Neurphys ('nervous')
--------------------
IO and analysis package built to streamline and standardize
the data handling, analysis, and visualization of electrophysiology
and calcium imaging data
"""

__version__ = '0.0'

from .calcium import calc_ca_conc
from .membrane import calc_mem_prop
from .oscillation import create_epoch, epoch_hist, epoch_kde, epoch_pgram
from .pv_import import parse_xml, import_vr_csv, import_ls_csv, import_folder
from .read_abf import read_abf, keep_sweeps, drop_sweeps
from .utilities import baseline, find_peak, calc_decay, simple_smoothing
from .synaptics import analyze_current, calc_ppr
