import numpy as np
from duffingtools.theory.brownian_particle import power_spectral_density as psd
from duffingtools.data_analysis.helper_functions import get_max_psd_coherent_drive
import lmfit
import matplotlib.pyplot as plt
from uncertainties import ufloat, umath

from scipy.constants import Boltzmann as kB  # Boltzmann's constant
from scipy.constants import N_A  # Avogadro's constant
from scipy.constants import elementary_charge  # elementary charge

from scipy.signal import welch
pi = np.pi

from duffingtools.utils.read_write import save_fig

def fit_psd(fx, px, initial_guess=None, fixed_parameters=[], frequency_range=[],
                      verbose=False, method='leastsq',return_fig = False):
    """
    :param fx:
    :param px:
    :param initial_guess:
    :param fixed_parameters:
    :param frequency_range:
    :param verbose:
    :param method:
    :return:
    """


    fig = None  # just a placeholder for now
    if len(frequency_range)>0:
        assert len(frequency_range) == 2
        p = px[np.all(np.array([fx >= frequency_range[0], fx <= frequency_range[1]]), axis=0)]
        f = fx[np.all(np.array([fx >= frequency_range[0], fx <= frequency_range[1]]), axis=0)]
    else:
        f, p = fx, px

    # if (save_fig_path is not None) and (verbose is not False):
    #     print('WARNING: save_fig_path not None and verbose False. no image will be generated!')



    df = np.mean(np.diff(fx))  # frequency spacing


    # ======== find initial estimates ================================================================
    # ================================================================================================

    if initial_guess is not None:
        initial_params = initial_guess

    else:
        fo = f[np.argmax(p)]

        p1, f1 = p[0:np.argmax(p)], f[0:np.argmax(p)]
        f_halfmax_min = f1[p1 > np.max(p) / 2][0]

        p1, f1 = p[np.argmax(p):], f[np.argmax(p):]
        f_halfmax_max = f1[p1 > np.max(p) / 2][-1]

        initial_params = {'fo': fo, 'g': f_halfmax_max-f_halfmax_min, 'noise': np.sum(px)*df}

    if verbose:
        fig = plt.figure()
        plt.semilogy(f, p, 'o', color='midnightblue', alpha = 0.5, label = 'data')
        for f_halfmax in [f_halfmax_min, f_halfmax_max]:
            plt.plot([f_halfmax, f_halfmax], [min(p), max(p)], 'k--', alpha = 0.75)
        plt.plot(f, psd(f, **initial_params), 'g', label='initial guess')

    # # # ======== actual fit =========================================================================
    # # ================================================================================================
    #
    # create the model for amplitude fit
    gmodel = lmfit.Model(psd)
    params = gmodel.make_params(**initial_params)
    #
    # # set the boundaries for the parameters
    # assume that the initial guess is not off more than 20%
    params['fo'].set(min=initial_params['fo']*0.8)
    params['fo'].set(max=initial_params['fo']*1.2)

    # assume that the initial guess is not off more than a factor 2
    params['g'].set(min=initial_params['g']*0.5)
    params['g'].set(max=initial_params['g']*2)

    # assume that the initial guess is not off more than a factor 2
    params['noise'].set(min=initial_params['noise']*0.5)
    params['noise'].set(max=initial_params['noise']*2)
    # #
    for var in fixed_parameters:
        if var in params:
            params[var].set(vary=False)

    # actually fit the model
    model_fit = gmodel.fit(p, params, f=f, method=method)

    if verbose:
        print('\n====== result of psd fit ======')
        print(model_fit.fit_report())  # print the model fit report

        p_fit = psd(f/1000, **model_fit.best_values)
        plt.plot(f/1000, p_fit, 'r-', label='fit')

        plt.legend()
        plt.xlabel(r'$\omega/2\pi$ (kHz)')
        plt.ylabel(r'$S_v$ (bit$^2$/Hz)')

        # if save_fig_path:
        #     save_fig(fig, save_fig_path)

    if return_fig:
        return model_fit, fig
    else:
        return model_fit


# def get_Rs()

def remove_peak(p, n_peak=2, verbose=False):
    """
    assumes that there is one prominent peak in p and removes it by replacing the 2*n_peak+1 values around the peak
    with the mean of p of the adjacent 2*n_peak values

    WARNING: this function actually changes the value of the input p!
    :param p: array with single strong peak
    :param n_peak: size over which to remove peak
    :param verbose: if true plot the data
    :return:
    """
    i_max = np.argmax(p)
    # range of peak
    rp = np.arange(i_max - n_peak, i_max + n_peak)
    # range of background
    rb = np.hstack([np.arange(i_max - 2 * n_peak, i_max - n_peak), np.arange(i_max + n_peak, i_max + 2 * n_peak)])

    if verbose:
        plt.figure()
        plt.plot(p[rp], 'o', label = 'peak data')
        plt.plot(p[rb], 'o', label= 'adjecent region')
        plt.plot(np.ones(len(rp)) * np.mean(p[rb]), '-', label = 'replacement')

        plt.legend()
    pn = p
    p_peak = p[rp]  # peak values
    pn[rp] = np.ones(len(rp)) * np.mean(p[rb])

    return rp, p_peak

def get_physical_params_from_psd_model(model_fit, Pgas, T = 300, rho_m=2200, Mgas=28.97e-3):
    """

    retrieve the physical parameters from the fitted models to power spectral density

    :param model_fit:
    :param Pgas: gas pressure in mBar
    :param T: temperature of environment in Kelvins,  default is 300K
    :param rho_m: mass density in units of kg/m^3, default is 2200kg/m^3
    :param Mgas: molar mass of molecules, default is 28.97e-3 kg/mol for dry air
    :return:
    """

    # get all the a parameters with their uncertainties
    fo, g, noise = [ufloat(m, s) for m, s in zip(model_fit.best_values.values(), np.sqrt(model_fit.covar.diagonal()))]

    wo = 2*pi*fo
    gamma = 2*pi*g
    mgas = Mgas/N_A

    # calculate the physical values with errors
    radius = 2.223 / rho_m * umath.sqrt(mgas / (kB * T)) * (Pgas*100) / gamma # radius in meters
    mass = 4*pi/3*rho_m*radius**3 # mass in kg

    C = wo**2  * noise / (T)  # energy calibration factor in (bit*Hz)^2 / Kelvin
    c = umath.sqrt(mass*wo**2*noise / (kB * T))*1e-9  # position calibration factor in bit/nm (or V/nm)



    return {'radius': radius*1e9, 'mass': mass, 'C': C, 'c': c, 'fo':fo, 'g':g, 'x2':noise*C}


def get_calibration_and_mass(x, fs,frequency_range=None, T=300, pressure=None, coherent_drive=False,
                             n_max=1e6, nq=1, Eo=None, return_fig = False, verbose=True):
    """

    run calibration and return the parameters in a dictionary

    the expected time trace is a thermal peak of the frequency range and a coherent drive which corresponds
    to the highest point in the spectra if



    :param x: time trace
    :param fs: sampling frequency
    :param frequency_range: frequency range for fit of thermal peak
    :param T: ambient temperature in Kelvins (default 300K)
    :param pressure: pressure at which measurement was taken in mBar
    :param coherent_drive: if true signal contains a coherent drive tone
    :param n_max: max length of timetrace to use for determination of coherent peak height (becomes very memory intensive for long measurements)
    :param nq: number of charges on particle
    :param Eo: electric field amplitude in V/m

    :param save_fig_path: if verbose on path to which to save the image
    :param verbose: if True print information and plots as script runs
    :return:
    """

    figs = [None, None]  # just a placeholder for now

    Qp = nq * elementary_charge

    f, p = welch(x, fs=fs, window='hanning', nperseg=2 ** 12, noverlap=None, nfft=None,
                 detrend='constant', return_onesided=True, scaling='density', axis=-1)

    fo, po = f[np.argmax(p)], p[np.argmax(p)]

    # for thermal fit remove the coherent drive if there is one
    if coherent_drive:
        rb, p_peak= remove_peak(p, n_peak=2, verbose=False)
        f_peak = f[rb]

    if verbose:
        print('max peak and power', fo, po / (2 ** 12) * fs)

    fit, figs[0] = fit_psd(f, p, frequency_range=frequency_range, verbose=verbose,
                  return_fig=return_fig)


    if verbose and return_fig and coherent_drive:
        # add the data for the peak to the plot
        plt.plot(f_peak, p_peak, 'o', label='removed peak')
        plt.legend()

    params = get_physical_params_from_psd_model(fit, Pgas=pressure)  # parameters from fit

    params_dict = {}
    params_dict.update({k: v.n for k, v in params.items()})
    params_dict.update({k + '_s': v.s for k, v in params.items()})

    # if save_fig_path:
    #     save_fig(fig, save_fig_path)

    # find the max of the coherent drive if there is one
    if coherent_drive:
        # we get the coherent drive amplitude but to reduce the memory requirements we just use max 1 million points
        fd, pd, figs[1] = get_max_psd_coherent_drive(x[0:min([len(x), int(n_max)])], fs, n_r=1, n_pts=20, n_it=2,
                                                     verbose=verbose, return_fig=return_fig)
        if verbose:
            print('max peak and power (2)', fd, pd / min([len(x), int(n_max)]) * fs)
        tau = min([len(x), int(n_max)]) / fs

        # thermal peak at drive frequency
        p_th = psd(fd, **fit.best_values)
        Rs = (pd - p_th) / p_th
        gamma0 = 2 * pi * fit.best_values['g']

        mass = Qp ** 2 * Eo ** 2 * tau / (8 * kB * T * gamma0 * Rs)
        params_dict.update({'mass (drive)': mass, 'Rs': Rs, 'pd': pd, 'psd_th': p_th})

    if return_fig:
        return params_dict, figs
    else:
        return params_dict