import numpy as np
import lmfit
import matplotlib.pyplot as plt
from uncertainties import ufloat, umath
import pandas as pd
import scipy
import scipy

pi = np.pi

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