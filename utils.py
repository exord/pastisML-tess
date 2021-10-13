"""
Utility functions, mostly taken from other packages.

Created on Fri May 28 17:58:22 2021

@author: rodrigo
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import pi
import numpy as np

#from . import constants as cts
import constants as cts


def sma(per, m1, m2):
    """
    Compute semi-major axis of relative orbit binary system.

    Primary star has mass m1[msun] and secondary has m2[msun]
    and period per [days].
    """
    per_s = per * 24 * 3600
    return (per_s**2 * (m1 + m2) * cts.GMsun / (4 * pi**2))**(1./3.) / cts.au


def r_infconj(ecc, omega_deg, ar):
    """
    Compute separation at the time of inferior conjunction (transit).

    Return value normalised to stellar radius

    :param float or np.array ecc: orbital eccentricity
    :param float or np.array omega_deg: argument of pericenter (in deg)
    :param float or np.array ar: semi-major axis of orbit in units of stellar
    radius
    """
    omega_rad = omega_deg * pi / 180.0
    return ar * (1 - ecc**2) / (1 + ecc * np.cos(pi/2 - omega_rad))
