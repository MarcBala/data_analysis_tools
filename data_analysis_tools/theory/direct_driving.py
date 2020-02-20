#!/usr/bin/python
# created: October 22th 2019
# author: Jan Gieseler
# modified: October 22th 2019
import numpy as np

pi = np.pi

# typical parameters
physical_parameters = {
    'Q_factor':1e8,
    'duffing_term': -10, 'nonlinear_damping': 9.5, 'f_0': 120, 'drive_strength': 1
}
parameter_units = {
    'force': r'aN', 'duffing_term': r'$\mu m^{-2}$', 'nonlinear_damping': r'$\mu m^{-2}$',
    'mass': r'fg', 'f_0': 'Hz', 'drive_strength': 'm/s^2'
}


# def phase_response(phase, drive_strength, nonlinear_damping, duffing_term, f_0, Q_factor=1e5, phi0=0):
#     """
#
#     solution to the Duffing equation with nonlinear damping when phase is the free parameter
#     for direct force
#     returns amplitude and frequency
#
#     phase in deg
#     drive_strength: unitless will be converted to an acceleration, i.e. the force divided by the mass
#     eta: feedback gain in 1/um^2
#     xi: feedback gain in 1/um^2
#     f0: frequency in Hz
#     """
#
#     phi = (phase - phi0) * pi / 180
#
#     wo = 2*pi*f_0
#
#     g = drive_strength * Q_factor**(3/2) * duffing_term / wo**2
#     print('g', g)
#     r2 = ( g**2 / (4*wo**4 * nonlinear_damping**2) * (np.sin(phi))**2 )**(1/3)
#
#     r2[r2 < 0] = np.nan
#     print(duffing_term)
#     f = f_0 * (1 -  g * np.sqrt(np.abs(duffing_term)) / (4 * wo**2 * np.sqrt(r2)) * np.cos(phi) + 3 / 2 * duffing_term * r2)
#
#     return r2, f

def physical2cparams(drive_strength, nonlinear_damping, duffing_term, f_0, Q_factor=None):
    wo = 2 * pi * f_0

    c0 = nonlinear_damping**(-1/3) * (drive_strength / (2*wo**2) )**(1/3)

    c1 = 0  # this is an experimental delay, which in the ideal experiment is zero

    c2 = f_0

    # c3 = -nonlinear_damping**(1/3)/ 2 * (drive_strength / (2*wo**2))**(2/3) * f_0
    # c4 = 3 * duffing_term * nonlinear_damping**(-2/3) / 2 * (drive_strength / (2*wo**2))**(2 / 3) * f_0


    alpha = 3*duffing_term/nonlinear_damping
    c3 = -0.5*(drive_strength/(2*wo**2))**(2/3)*nonlinear_damping**(1/3)*wo*np.sqrt(1+alpha**2)
    c4 = np.arctan2(3 * duffing_term, nonlinear_damping) * 180 / pi

    return c0, c1, c2, c3, c4



def phase_response(phase, drive_strength, nonlinear_damping, duffing_term, f_0, Q_factor=1e5, phi0=0, assume_infinite_Q =True):
    """

    solution to the Duffing equation with nonlinear damping when phase is the free parameter
    for direct force
    returns amplitude and frequency

    phase in deg
    drive_strength: this is an acceleration, i.e. the force divided by the mass
    eta: feedback gain in 1/um^2
    xi: feedback gain in 1/um^2
    f0: frequency in Hz
    """

    wo = 2 * pi * f_0

    g = drive_strength * Q_factor ** (3 / 2) * np.sqrt(np.abs(duffing_term)) / wo ** 2


    phi = (phase - phi0) * pi / 180

    # NEED TO CHECK FOLLOWING 05.11.2019
    # if the duffing term is negative, we get this additional phase factor from the sqrt of -1
    # if duffing_term<0:
    #     phi += pi/2

    A = 3/4 * nonlinear_damping / abs(duffing_term)
    B = 27/32 * g * (nonlinear_damping / duffing_term)**2 * np.sin(phi)
    C = (B + np.sqrt(np.abs(4 * A ** 3 + B ** 2))) ** (1 / 3)

    if assume_infinite_Q: # use the simpler fomular for infinite Q
        r = (drive_strength / (2 * wo ** 2 * nonlinear_damping) * np.sin(phi)) ** (1 / 3)
    else:
        r = 2 / (3 * nonlinear_damping) * np.sqrt(abs(duffing_term) / Q_factor) * (C - A / C)

    r[r<0] = np.nan # negative amplitudes are not physical
    r2 = r ** 2

    w = wo * (1 - drive_strength / (4 * wo ** 2 * r) * np.cos(phi) + 3 / 2 * duffing_term * r2)
    f = w/(2*np.pi)

    return r2, f


def phase_response_ampl(phase, c0, c1, amplitude_squared=True):
    """

        amplitude response direct drive

    """

    phi = (phase+c1)
    # there is only a valid response for 0<phase<180, any additional shifts should be taken care of by c1
    assert min(phi) > 0, 'the minimum allowable phase is 180 (actual max is %0.2f)' % float(min(phi))
    assert max(phi) < 180, 'the maximum allowable phase is 180 (actual max is %0.2f)' % float(max(phi))

    phi = phi * np.pi / 180  # convert to rads

    r = c0 * (np.sin(phi))**(1/3)

    if amplitude_squared:
        return r**2
    else:
        return r


def phase_response_freq(phase, c2, c3, c4):
    """

        frequency response direct drive

    """

    # there is only a valid response for 0<phase<180, any additional shifts should be taken care of before calling this function
    assert min(phase) > 0
    assert max(phase) < 180

    phi = phase*np.pi/180

    f = c2 + c3 / np.sin(phi)**(1 / 3) * np.cos(phi + c4 * np.pi / 180)

    return f




if __name__ == '__main__':
    phase = np.arange(0,3.1, 0.25)*180/pi
    r, f, r1 = phase_response(phase, **physical_parameters)
    print('r:', r)
    print('f:',f)