import numpy as np
import lmfit
import matplotlib.pyplot as plt
from uncertainties import ufloat, umath
import pandas as pd
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


def load_and_calc_psd_map(folder, time_step=1 / 625e3, channel=1, Navg=50, max_time=0.1, freq_range=[100e3, 150e3],
                          verbose=False, Ccal=2e4):
    """

    loads data from folder and plots the 2D map

    time_step: time step in seconds

    max_time: max time duration of timetrace to cut the time trace to speed up calculation

    freq_range: freq range of interst
    """

    # get the filenames sorted by the first index
    filenames = sorted([f for f in (folder).glob('*.bin')], key=lambda x: int(x.name.split('_')[0]))
    if verbose:
        print('number of files found: ', len(filenames))
    psd_map = []
    for filename in filenames:

        data = load_timetrace_binary(filename, time_step=time_step)
        if max_time is None:
            x = data[channel]
        else:
            x = data[channel][0:int(max_time / time_step)]
        f, p = power_spectral_density(x, time_step, freq_range=freq_range, N_pieces=Navg)
        psd_map.append(p / Ccal ** 2)
    psd_map = np.array(psd_map)

    if verbose:
        print('shapes of f and psd_map: ', f.shape, psd_map.shape)

    return f, psd_map


def find_extema(x, window_size=21, polynomial_order=3, max_filter_iterations=10, max_change=1e-3, verbose=False):
    """
    find extrema of curves by first smoothing input data x and then finding the extrema of the smooth curve

    returns the index of the global min and max of x

    note that currently this works only to find a single value for min max, which is the global min/max ignoring the borders

    we use a repeatedly applied savgol_filter until the data is sufficiently smooth

    the data is considered smooth if the change from one smoothing iteration to the next is less than max_change
    or max number of iterations has been reached

    x: input data

    window_size: windowsize of savgol filter
    polynomial_order: polynomial_order of savgol filter
    max_filter_iterations: stop critera - max iterations
    max_change: stop critera - if change is less than maximum_change stop filtering




    """

    #    kind: kind of extrema to look for options are 'min', 'max', 'min_max'


    def change(y1, y2):
        err, val_range = np.sqrt(np.mean((y1 - y2) ** 2)) / 2, np.max(y1) - np.min(y1)
        return err / val_range

    xs = x  # xs is the smoothed version of x
    if verbose:
        print(' ----- smoothing data ----- ')
    for i in range(max_filter_iterations + 1):
        xs_old = xs
        xs = savgol_filter(xs, window_size, polynomial_order)  # apply filter

        if change(xs, xs_old) < max_change:
            break
        c = change(xs, xs_old)

        if verbose:
            print(c)

    if i == max_filter_iterations:
        print('WARNING: max_change not reached')

    peaks_min, _ = find_peaks(-xs, distance=len(x))

    peaks_max, _ = find_peaks(xs, distance=len(x))

    if verbose:
        print(peaks_max, len(xs))
        plt.figure()
        plt.plot(x)
        plt.plot(xs)
        plt.plot(peaks_min, xs[peaks_min], 'o', alpha=0.8, markersize=10, label = 'min')
        plt.plot(peaks_max, xs[peaks_max], 'o', alpha=0.8, markersize=10, label = 'max')
        plt.legend()

    return peaks_min, peaks_max


def get_max_psd_coherent_drive(qx, fs, n_r=1, n_pts=10, n_it=2, verbose=False, return_fig=True):
    """
    get the maximum of the power spectral density of a coherent drive, note that since this is a delta function the
    total power in the peak is po*df with df = fs/len(qx)
    :param qx: timetrace with signature of coherent drive
    :param fs: sampling rate (Hz)
    :param n_r: range of fourier pts for optimization
    :param n_pts: number of points used in numerical optimization, the more the better the resolution but also the longer the calculation
    :param n_it: number iterations used in numerical optimization, the more the better the resolution but also the longer the calculation
    :param verbose: if verbose plot a graph that shows the psd in each iteration
    :return: fo and po, the frequency of the coherent drive and the power spectral density at this frequency
    """

    fig = None  # just a placeholder for now

    # caculate the FFT as a starting point
    f, p = welch(qx, fs=fs, window='boxcar', nperseg=len(qx), noverlap=None, nfft=None, detrend='constant',
                 return_onesided=True, scaling='density', axis=-1)

    for i in range(n_it):
        # get the peak and freq
        fo = f[np.argmax(p)]
        po = p[np.argmax(p)]

        # range is over three freq points of previous iteration
        r = np.arange(np.argmax(p) - n_r, np.argmax(p) + n_r + 1)
        w = 2 * pi * np.linspace(f[r][0], f[r][-1], n_pts)
        if verbose and i == 0 and return_fig:
            fig, ax = plt.subplots(1, 1)
            ax.plot(f[r], p[r], 'o', markersize=8, alpha=0.5, label='FFT')
            ax.plot(fo, po, 'x', markersize=10, label='max (FFT)')

        A = A_fun(qx, w, dt=1 / fs, n_max=None)
        f, p = w / (2 * pi), 2 * np.abs(A) ** 2 / fs * len(qx)

        # get the peak and freq
        fo = f[np.argmax(p)]
        po = p[np.argmax(p)]

        if verbose and return_fig:
            print(fo, po)
            ax.plot(f, p, 'o', markersize=5, alpha=0.5, label='iter {:0d}'.format(i + 1))
    if verbose and return_fig:
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('PSD ($V^2$/Hz)')
        ax.legend()
        # if save_fig_path:
        #     save_fig(fig, save_fig_path)

    if return_fig:
        return fo, po, fig
    else:
        return fo, po


# --------------------------------------------------------------------------------------------------------------------
# ========== A_fun ===================================================================================================
# --------------------------------------------------------------------------------------------------------------------
def A_fun(qx, w, dt, n_max=None):
    '''
    Ak = A_fun(qx, w, fs)
    input:
        qx: input signal vector length N
        w: omega, w = 2*pi*k*fs / M  vector length K
        dt: sampling interval
        n_max:(optional) batch size when calculating the integral (sum), necessary for large arrays qx due to memory problem
            use something like 1e4
    output:
        Ak: spectrum at frequencies w to get to the single sided power spectral density calculate PSD = 2*abs(A)**2*dt*len(qx)
    '''

    if n_max is None:
        n_max = int(len(qx))

    N = len(qx)
    j = 1j
    n_max = int(n_max)  # make sure its an integer

    if n_max < N:
        Ak = np.zeros(len(w)) + j
        for k in range(int(N / n_max)):
            #             print(k, k*n_max, (k+1)*n_max)
            nn = np.array([np.arange(k * n_max, (k + 1) * n_max)]).T
            qk = qx[k * n_max:(k + 1) * n_max]

            Ak += np.dot(np.array([qk]), np.exp(-j * np.dot(nn, np.array([w])) * dt))[0]

        Ak = Ak / N
        return Ak
    else:

        nn = np.array([np.arange(N)]).T

        Ak = (1. / N) * np.dot(np.array([qx]), np.exp(-j * np.dot(nn, np.array([w])) * dt))

        return Ak[0]

