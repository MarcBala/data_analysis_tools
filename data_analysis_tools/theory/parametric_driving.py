#!/usr/bin/python
# created: September 25th 2019
# author: Jan Gieseler
# modified: October 3rd 2019
import numpy as np

pi = np.pi

# typical parameters
physical_parameters = {
    'Q_factor': 1e5, 'duffing_term': -10, 'nonlinear_damping': 9.5, 'f_0': 120, 'mod_depth': 4
}
parameter_units = {
    'Q_factor': '', 'force': r'aN', 'duffing_term': r'$\mu m^{-2}$', 'nonlinear_damping': r'$\mu m^{-2}$',
    'mass': r'fg', 'f_0': 'Hz', 'mod_depth': '%'
}


def phase_response(phase, mod_depth, nonlinear_damping, duffing_term, f_0, Q_factor=1e10, phi0=0):
    """

    solution to the Duffing equation with nonlinear damping when phase is the free parameter
    returns amplitude and frequency

    phase in deg
    epsilon: modulation depth in %
    eta: feedback gain in 1/um^2
    xi: feedback gain in 1/um^2
    f0: frequency in Hz
    Q: Q-factor (no units)
    """

    eps = 0.01 * mod_depth
    phi = (phase - phi0) * np.pi / 180

    r2 = -2 * eps / nonlinear_damping * np.sin(phi) - 4 / (nonlinear_damping * Q_factor)

    r2[r2 < 0] = np.nan

    f = f_0 * (1 + eps / 4 * np.cos(phi) + 3 / 8 * duffing_term * r2)

    return r2, f


def freq_response(frequency, mod_depth, nonlinear_damping, duffing_term, f_0, Q_factor):
    """

    solution to the Duffing equation with nonlinear damping when frequency is the free parameter
    returns amplitude and phase

    phase in deg
    epsilon: modulation depth in %
    eta: feedback gain in 1/um^2
    xi: feedback gain in 1/um^2
    f0: frequency in Hz
    Q: Q-factor (no units)
    """

    delta_m = 2 - frequency / f_0
    eps = 0.01 * mod_depth
    alpha = 3 * duffing_term / nonlinear_damping
    delta_th = np.sqrt(1 + alpha ** 2) / 2
    r2 = -1 / (nonlinear_damping * delta_th ** 2) * (alpha * delta_m + 1 / Q_factor - np.sqrt(
        eps ** 2 * delta_th ** 2 - delta_m ** 2 + alpha / Q_factor ** 2 * (2 * Q_factor * delta_m - alpha)))

    r2[r2 < 0] = np.nan

    phi = 0.5 * np.arccos(-2 / eps * delta_m - 3 / (2 * eps) * duffing_term * r2)

    return r2, phi


def phase_response_ampl(phase, c0, c1):
    """

        amplitude response

    """

    phi = phase*np.pi/180
    r2 = c0 * np.cos(phi+c1*np.pi/180)
    return r2


def phase_response_freq(phase, c2, c3, c4):
    """

        frequency response

    """

    phi = phase*np.pi/180
    f = c2 + c3 * np.cos(phi+c4*np.pi/180)
    return f

