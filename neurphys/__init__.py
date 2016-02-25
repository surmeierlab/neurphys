"""
Neurphys ('nervous')
--------------------
IO and analysis package built to streamline and standardize
the data handling, analysis, and visualization of electrophysiology
and calcium imaging data
"""

__version__ = '0.0'

from .utilities import baseline, find_peak, calc_decay, simple_smoothing
from .calcium import calc_ca_conc
from .membrane import calc_mem_prop
from .nuplot import simple_axis, simple_figure, clean_axis, clean_figure, nu_legend, nu_boxplot, nu_scatter, nu_raster, nu_violin
from .oscillation import create_epoch, epoch_hist, epoch_kde, epoch_pgram
from .pacemaking import detect_peaks, baseline_pacemaking, calc_freq
from .read_abf import read_abf, keep_sweeps, drop_sweeps
from .read_pv import parse_xml, import_vr_csv, import_ls_csv, import_folder
from .synaptics import analyze_current, calc_ppr
