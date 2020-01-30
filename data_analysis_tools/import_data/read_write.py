#!/usr/bin/python
# created: January 30th 2020
# author: Marc Torrent
# modified:

import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
# %precision 3 # precission when printing
import matplotlib
# from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import LogFormatter, ScalarFormatter, StrMethodFormatter

FONT_SIZE = 12
MARKER_SIZE = 4
matplotlib.rc('xtick', labelsize=FONT_SIZE)
matplotlib.rc('ytick', labelsize=FONT_SIZE)
matplotlib.rc('axes', titlesize=10)
matplotlib.rc('axes', labelsize=FONT_SIZE)
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=MARKER_SIZE)

matplotlib.rc('legend', fontsize=FONT_SIZE)
matplotlib.rc('legend', handlelength=2)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patches as patches

import matplotlib.gridspec as gridspec
import yaml

from matplotlib import ticker, cm

pi = np.pi


def save_fig(fig, filename, **kwargs):
    """
    wrapper for save fig, that checks and creates the required folder

    """

    folder_path = filename.parent

    paths = []
    while not folder_path.exists():
        paths.append(folder_path)
        folder_path = folder_path.parent
    for path in paths[::-1]:
        print('make directory:', path)
        path.mkdir()

    fig.savefig(filename, **kwargs)


def load_timetrace_binary(filename, time_step=1 / 625e3, skip_time=0.25, total_time=5, N_channels=4, verbose=False):
    """
    load the binary file from the Labview acquisition

    """

    #     TimeTrace_4Ch_10s_Vpdr=0.310V_Phase=0
    #     info = {k:v for v, k in zip(filename.name.split('.bin')[0].split('_'), ['id', 'Name', 'channels', 'duration', 'eta', 'drive'])}
    #     info['channels'] = int(info['channels'].split('Ch')[0])
    #     info['duration'] = int(info['duration'].split('s')[0])
    #     info['drive'] = int(info['drive'].split('Vpdr=')[1].split('mV')[0])

    #     N_channels = info['channels']
    data = pd.DataFrame(np.fromfile(str(filename), dtype=np.int16)).values[N_channels:, 0]  # the first entries are all zeros
    data = data.reshape(-1, N_channels)


    N_skip = int(skip_time / time_step)
    N_final = int((total_time + skip_time) / time_step)

    return data[N_skip:N_final, 0:4].T  # the last channel doesn't contain data


#     return info, data[N_skip:N_final, 0:3].T # the last channel doesn't contain data
#     return info, data.T # the last channel doesn't contain data


def load_ZI_sweep(filename):
    drive_voltage = filename.name.split('_Vpdr=')[1].split('V')[0]

    data = pd.read_csv(filename, index_col=0)
    data['R'] = np.sqrt(data['X_m'] ** 2 + data['Y_m'] ** 2) * np.sqrt(2) * (
                32768 / 10)  # convert rms to amplitude in bits

    return data



