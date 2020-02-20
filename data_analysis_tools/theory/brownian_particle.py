import numpy as np

pi = np.pi

def power_spectral_density(f, fo, g, noise):
    """

    theoretical power spectral density as defined in Advances in Optics & Photonics Eq.

    frequencies are regular frequencies not angular frequencies


    :param f: frequency in Hz
    :param fo: resonance frequencies
    :param g: damping g = gamma/2 pi
    :param noise: noise amplitude corresponding to the variance of the position of a Brownian particle
                  in a harmomic trap at temperature T noise = <x^2> = kB T / (m wo^2), with wo = 2*pi*fo
    :return: power spectral density
    """


    return 2*noise/pi * fo**2*g / ((f**2-fo**2)**2 + f**2*g**2)