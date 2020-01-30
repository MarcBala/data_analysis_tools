#!/usr/bin/python
# created: January 30th 2020
# author: Marc Torrent
# modified:

import numpy as np
import pandas as pd

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

    return data[N_skip:N_final, 0:4].T  # 0:3 3ch /// 0:4 4Ch---


#     return info, data[N_skip:N_final, 0:3].T # the last channel doesn't contain data
#     return info, data.T # the last channel doesn't contain data


def load_ZI_sweep(filename):
    drive_voltage = filename.name.split('_Vpdr=')[1].split('V')[0]

    data = pd.read_csv(filename, index_col=0)
    data['R'] = np.sqrt(data['X_m'] ** 2 + data['Y_m'] ** 2) * np.sqrt(2) * (
                32768 / 10)  # convert rms to amplitude in bits

    return data

def power_spectral_density(x, time_step, freq_range=None, N_pieces=None):
    """
    returns the *single sided* power spectral density of the time trace x which is sampled at intervals time_step


    gives the same result as scipy.scipy.signal where N_piece = len(x) / nperseg and window = 'boxcar'

    Args:
        x (array):  timetrace
        time_step (float): sampling interval of x
        freq_range (array or tuple): frequency range in the form [f_min, f_max] to return only the spectrum within this range
        N_pieces: if not None should be integer and the timetrace will be chopped into N_pieces parts, the PSD calculated for each and the avrg PSD is returned
    Returns:

    """
    if N_pieces is not None:
        assert type(N_pieces) is int
        F, P = [], []
        for x_sub in np.reshape(x[0:int(len(x) / N_pieces) * N_pieces], (N_pieces, int(len(x) / N_pieces))):
            F_sub, P_sub = power_spectral_density(x_sub, time_step, freq_range=freq_range, N_pieces=None)
            F.append(F_sub)
            P.append(P_sub)
        F = np.mean(F, axis=0)
        P = np.mean(P, axis=0)
    else:
        N = len(x)
        P = 2 * np.abs(np.fft.rfft(x)) ** 2 / N * time_step
        F = np.fft.rfftfreq(len(x), time_step)

        if freq_range is not None:
            brange = np.all([F >= freq_range[0], F <= freq_range[1]], axis=0)
            P = P[brange]
            F = F[brange]

    return F, P

