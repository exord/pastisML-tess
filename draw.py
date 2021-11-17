"""
This module contains functions to randomly draw FP or planet system parameters.

Created on Wed Feb 17 17:23:07 2021

@author: rodrigo
"""
# -*- coding: utf-8 -*-

import os
import numpy as np
# import pandas as pd
'''
from . import core as c
from . import constants as cts
from . import parameters as p
from . import utils as u
'''
import core as c
import constants as cts
import parameters as p
import utils as u



homedir = os.getenv('HOME')
mldir = os.path.join(homedir, 'EXOML', 'TESS-pastis')


def draw_parameters(params, scenario, nsimu=1, **kwargs):
    """
    Draw the parameters for a set of target (TIC) star.

    The draw is conditional on the parameter sets, representing the TIC star.

    :param iterable params: shape (nparams, size), list of parameters of TIC
    star. Is the second dimension is > 1, then nsimu draws are performed for
    each parameter set.

    :param str scenario: defines the scenario for which to draw parameters

    :param int nsimu: size of parameter set to draw

    The remaining kwargs are passed to the _draw_parameters functions
    """
    ticstar = c.TargetStarParameters(params)
    ticstar.draw()

    # Draw parameters and return input dict for pastis
    if scenario.lower() in ['pla', 'planet']:
        planet = _draw_parameters_pla(ticstar, **kwargs)

        # Flag non-transiting planets
        flag = conservative_transit_flag([ticstar, planet])

        # Construct planet dict for pastis
        # Add tree-like structure
        planetdict = {'star1': 'Target1', 'planet1': 'Planet1'}

        input_dict = {'Target1': ticstar.to_pastis(flag),
                      'Planet1': planet.to_pastis(flag),
                      'PlanSys1': planetdict}

    elif scenario.lower() == 'beb':
        beb_params = _draw_parameters_beb(ticstar, **kwargs)

        # Flag non-transiting planets
        flag = conservative_transit_flag(beb_params)

        # Construct binary dict for pastis
        binarydict = beb_params[-1].to_pastis(flag)
        # Add tree-like structure
        binarydict.update({'star1': 'Blend1', 'star2': 'Blend2'})

        # binarydict['P'] = beb_params[1].period

        tripledict = _draw_parameters_boundbinary(ticstar, **kwargs)
        
        tripledict.update({'object1': 'Target1',
                           'object2': 'IsoBinary1'})

        input_dict = {'Target1': ticstar.to_pastis(flag),
                      'Blend1': beb_params[0].to_pastis(flag),
                      'Blend2': beb_params[1].to_pastis(flag),
                      'IsoBinary1': binarydict,
                      'Triple1': tripledict
                      }
        
    elif scenario.lower() == 'triple':
        bbinary_params = _draw_parameters_boundbinary(ticstar, **kwargs)
        
        # Flag non-transiting planets
        flag = conservative_transit_flag(bbinary_params)

        # Construct binary dict for pastis
        binarydict = bbinary_params[-1].to_pastis(flag)
        # Add tree-like structure
        binarydict.update({'star1': 'Blend1', 'star2': 'Blend2'})

        # binarydict['P'] = beb_params[1].period
        hierch_orbit = c.OrbitParameters(orbittype='triple')
        hierch_orbit.draw(sum(flag))
        
        triple_dict = hierch_orbit.to_pastis()
        triple_dict.update({'object1': 'Target1', 
                            'object2': 'IsoBinary1'})
        
        
        input_dict = {'Target1': ticstar.to_pastis(flag),
                      'Blend1': bbinary_params[0].to_pastis(flag),
                      'Blend2': bbinary_params[1].to_pastis(flag),
                      'IsoBinary1': binarydict,
                      'Triple1': tripledict
                      }
        

    return input_dict, flag


def _draw_parameters_pla(ticstar, **kwargs):
    """Draw parameters for the Planet scenario."""
    minimum_radius = kwargs.pop('minradius', p.MIN_PLA_RADIUS)
    # Instatiate planetary parameters
    planet = c.PlanetParameters(minradius=minimum_radius)
    # Draw parameters
    planet.draw(len(ticstar))

    return planet


def _draw_parameters_beb(ticstar, **kwargs):
    """
    Draw parameters for the BEB scenario.

    This is done by drawing a background star and build the binary, much like
    the pastis object builder.
    """
    # Build primary
    bkg_primary = c.BackgroundStarParameters(ticstar, minmass=0.5,
                                             maxdist=5)
    # Draw parameters for primary
    bkg_primary.draw(len(ticstar))

    # Build secondary
    bkg_secondary = c.SecondaryStarParameters(bkg_primary)
    bkg_secondary.draw()

    # Draw orbit
    orbit = c.OrbitParameters(orbittype='binary')
    orbit.draw(len(ticstar))

    return [bkg_primary, bkg_secondary, orbit]


def _draw_parameters_boundbinary(ticstar, **kwargs):
    """
    Draw parameters for the binary bound to a Target star.

    This is done by drawing a background star and build the binary, much like
    the pastis object builder.
    """
    # Build primary
    binary_primary = c.BoundPrimaryParameters(ticstar, minmass=0.5)
    # Draw parameters for primary
    binary_primary.draw(len(ticstar))

    # Build secondary
    binary_secondary = c.SecondaryStarParameters(binary_primary)
    binary_secondary.draw()

    # Draw orbit
    orbit = c.OrbitParameters(orbittype='binary')
    orbit.draw(len(ticstar))

    return [binary_primary, binary_secondary, orbit]


def conservative_transit_flag(params):
    """
    Flag systems according to whether they transit or not.

    Use a conservative approach when selecting stellar masses and radii.
    The objective at this point is performing a cut without having to buld the
    actual pastis objects.

    :param (Parameter class) params: instance of the parameter Class
    """
    # Check which class input belongs to.
    # Target + planet
    if (isinstance(params[0], c.TargetStarParameters) and
            isinstance(params[1], c.PlanetParameters)):
        # do something planety
        print('Checking parameters for planetary system')

        # Get masses
        mass2 = params[1].mass_mearth * cts.GMearth / cts.GMsun
        radius2_au = params[1].radius_rearth * cts.Rearth / cts.au

        # To be concervative, choose the smallest reasonable mass
        # and the largest possible radius
        # This will make the planet orbit closer to a larger star
        mass1 = 0.1
        radius1_au = 10.0 * cts.Rsun / cts.au

        # Define object containing orbital parameters
        orbit_params = params[1]

    # background star + secondary
    elif (isinstance(params[0], c.BackgroundStarParameters) and
          isinstance(params[1], c.SecondaryStarParameters)):
        # do something BEB
        print('Checking parameters for BEB system')

        assert len(params) > 2, "Missing parameter object for the orbit"

        # Get masses
        mass1 = params[0].mass
        mass2 = params[1].mass

        # Get radii
        # Again, to be conservative, choose LARGE radius
        radius1_au = 10.0 * cts.Rsun / cts.au
        radius2_au = 10.0 * cts.Rsun / cts.au

        # Define object containing orbital parameters
        orbit_params = params[2]

    # Get relevant orbital parameters
    periods = orbit_params.period
    ecc = orbit_params.ecc
    omega_deg = orbit_params.omega_deg
    incl_rad = orbit_params.incl_rad

    # Compute separation at inferior conjunction
    sma_au = u.sma(periods, mass1, mass2)
    r0 = u.r_infconj(ecc, omega_deg, sma_au / radius1_au)

    # compute impact parameter
    b = r0 * np.cos(incl_rad)

    # Return condition of transit
    return b <= 1 + radius2_au/radius1_au
