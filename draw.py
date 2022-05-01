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
        input_dict = {'Target1': ticstar.to_pastis(flag),
                      'Blend1': beb_params[0].to_pastis(flag),
                      'Blend2': beb_params[1].to_pastis(flag),
                      'IsoBinary1': binarydict,
                      }

    elif scenario.lower() == 'btp':
        # First, create planetary system
        plansys_params = _draw_parameters_bkgplansys(ticstar, **kwargs)

        # Flag non-transiting planets
        flag = conservative_transit_flag(plansys_params)

        # Construct planetary system dict for pastis
        plansysdict = {'star1': 'Blend1', 'planet1': 'Planet1'}

        # binarydict['P'] = beb_params[1].period
        input_dict = {'Target1': ticstar.to_pastis(flag),
                      'Blend1': plansys_params[0].to_pastis(flag),
                      'Planet1': plansys_params[1].to_pastis(flag),
                      'PlanSys1': plansysdict,
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

        tripledict = hierch_orbit.to_pastis()
        tripledict.update({'object1': 'Target1',
                           'object2': 'IsoBinary1'})


        input_dict = {'Target1': ticstar.to_pastis(flag),
                      'Blend1': bbinary_params[0].to_pastis(flag),
                      'Blend2': bbinary_params[1].to_pastis(flag),
                      'IsoBinary1': binarydict,
                      'Triple1': tripledict
                      }

    elif scenario.lower() == 'pib':
        # First, create planetary system
        plansys_params = _draw_parameters_boundplansys(ticstar, **kwargs)

        # Flag non-transiting planets
        flag = conservative_transit_flag(plansys_params)

        # Construct planetary system dict for pastis
        plansysdict = {'star1': 'Blend1', 'planet1': 'Planet1'}

        # Draw orbital parameters for triple
        hierch_orbit = c.OrbitParameters(orbittype='triple')
        hierch_orbit.draw(sum(flag))
        
        # Construct dictionary of hierarchichal orbit
        tripledict = hierch_orbit.to_pastis()
        tripledict.update({'object1': 'Target1',
                           'object2': 'PlanSys1'})

        # binarydict['P'] = beb_params[1].period
        input_dict = {'Target1': ticstar.to_pastis(flag),
                      'Blend1': plansys_params[0].to_pastis(flag),
                      'Planet1': plansys_params[1].to_pastis(flag),
                      'PlanSys1': plansysdict,
                      'Triple1': tripledict
                      }

    elif scenario.lower() == 'eb':
        # Draw parameters for secondary
        # need to add attribute to Target star
        ticstar.minmass = 0.0
        secondary_params = _draw_parameters_secondary(ticstar, **kwargs)

        # Build full binary
        eb_params = [ticstar, *secondary_params]

        # Flag non-transiting binaries
        flag = conservative_transit_flag(eb_params)

        # Construct binary dict for pastis
        binarydict = eb_params[-1].to_pastis(flag)

        # Update orbital parameters with values from secondary
        # This is because the qBinary class in PASTIS is not
        # written the same way as IsoBinary, for example; only
        # one star is required here
        binarydict.update(eb_params[1].to_pastis(flag))

        # Add tree-like structure
        # Only primary required in qBinary
        binarydict.update({'star1': 'Target1'})#, 'star2': 'Blend2'})

        # binarydict['P'] = beb_params[1].period
        input_dict = {'Target1': ticstar.to_pastis(flag),
                    #   'Blend2': eb_params[1].to_pastis(flag),
                      'qBinary1': binarydict,
                      }

    return input_dict, flag


def _draw_parameters_pla(ticstar, **kwargs):
    """Draw parameters for the Planet scenario."""
    
    # Minimum planetary radius
    minimum_radius = kwargs.pop('minradius', p.MIN_PLA_RADIUS)

    # Instatiate planetary parameters
    planet = c.PlanetParameters(minradius=minimum_radius, **kwargs)
    # Draw parameters
    planet.draw(len(ticstar), **kwargs)

    return planet


def _draw_parameters_beb(ticstar, **kwargs):
    """
    Draw parameters for the BEB scenario.

    This is done by drawing a background star and build the binary, much like
    the pastis object builder.
    """

    maxdist = kwargs.pop('maxdist', p.MAX_DIST)

    # Build primary
    bkg_primary = c.BackgroundStarParameters(ticstar, minmass=0.5,
                                             maxdist=maxdist)

    # TODO: this could be replaced with PrimaryBkgParameters (should be the same)

    # Draw parameters for primary
    bkg_primary.draw()

    # Build secondary
    bkg_secondary = c.SecondaryBkgParameters(bkg_primary)
    bkg_secondary.draw()

    # Draw orbit
    orbit = c.OrbitParameters(orbittype='binary', **kwargs)
    orbit.draw(len(ticstar), **kwargs)

    return [bkg_primary, bkg_secondary, orbit]


def _draw_parameters_bkgplansys(ticstar, **kwargs):
    """
    Draw parameters for a planetary system blended to a Target star.

    This is done by drawing a background star and build the system, much like
    the pastis object builder.
    """

    maxdist = kwargs.pop('maxdist', p.MAX_DIST)

    # Build planet host
    planet_host = c.BackgroundStarParameters(ticstar, minmass=0.5,
                                             maxdist=maxdist)
    # Draw parameters for planet host
    planet_host.draw()

    # Build planet
    minimum_radius = kwargs.pop('min_radius', p.MIN_DILUTED_PLANET_RADIUS)
    planet = _draw_parameters_pla(planet_host, minradius=minimum_radius, **kwargs)
    planet.draw(len(planet_host), **kwargs)

    # Draw orbit
    # orbit = c.OrbitParameters(orbittype='planet')
    # orbit.draw(len(ticstar))

    return [planet_host, planet]


def _draw_parameters_boundbinary(ticstar, **kwargs):
    """
    Draw parameters for the binary bound to a Target star.

    This is done by drawing a background star and build the binary, much like
    the pastis object builder.
    """
    # Build primary
    binary_primary = c.BoundPrimaryParameters(ticstar, minmass=0.5)
    # Draw parameters for primary
    binary_primary.draw()

    # Build secondary
    binary_secondary = c.SecondaryBkgParameters(binary_primary)
    binary_secondary.draw()

    # Draw orbit
    orbit = c.OrbitParameters(orbittype='binary', **kwargs)
    orbit.draw(len(ticstar), **kwargs)

    return [binary_primary, binary_secondary, orbit]


def _draw_parameters_boundplansys(ticstar, **kwargs):
    """
    Draw parameters for a planetary system bound to a Target star.

    This is done by drawing a bound star and build the system, much like
    the pastis object builder.
    """
    # Build planet host
    planet_host = c.BoundPrimaryParameters(ticstar, minmass=0.5)
    # Draw parameters for planet host
    planet_host.draw()

    # Build planet
    minimum_radius = kwargs.pop('min_radius', p.MIN_DILUTED_PLANET_RADIUS)
    planet = _draw_parameters_pla(planet_host, minradius=minimum_radius, **kwargs)
    planet.draw(len(planet_host), **kwargs)

    # Draw orbit
    # orbit = c.OrbitParameters(orbittype='planet')
    # orbit.draw(len(ticstar))

    return [planet_host, planet]


def _draw_parameters_secondary(ticstar, **kwargs):
    """
    Draw parameters for a secondary star bound to the main target star.
    """
    if not ticstar.drawn:
        ticstar.draw()

    # Because SecondaryStarParameters require ticstar to have a mass
    # attribute, we will do that...

    binary_secondary = c.SecondaryStarParameters(ticstar)
    binary_secondary.draw()

    # Draw orbit
    orbit = c.OrbitParameters(orbittype='binary', **kwargs)
    orbit.draw(len(ticstar), **kwargs)

    orbit.q = binary_secondary.q

    return [binary_secondary, orbit]


def conservative_transit_flag(params):
    """
    Flag systems according to whether they transit or not.

    Use a conservative approach when selecting stellar masses and radii.
    The objective at this point is performing a cut without having to buld the
    actual pastis objects.

    :param (Parameter class) params: instance of the parameter Class
    """
    #TODO merge as many if conditions as possible
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

    # background star + secondary (BEB)
    elif (isinstance(params[0], c.BackgroundStarParameters) and
          isinstance(params[1], c.SecondaryBkgParameters)):
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

    # Triple condition
    elif (isinstance(params[0], c.BoundPrimaryParameters) and
          isinstance(params[1], c.SecondaryBkgParameters)):
        # do something triple
        print('Checking parameters for Triple system')

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

    # PIB condition
    elif (isinstance(params[0], c.BoundPrimaryParameters) and
          isinstance(params[1], c.PlanetParameters)):
        
        # do something planety
        print('Checking parameters for PiB system')

        # Get stellar mass
        mass1 = params[0].mass
        # get stellar radius
        # Again, to be conservative, choose LARGE radius
        radius1_au = 10.0 * cts.Rsun / cts.au
        
        # Get planet mass and orbital distance
        mass2 = params[1].mass_mearth * cts.GMearth / cts.GMsun
        radius2_au = params[1].radius_rearth * cts.Rearth / cts.au

        orbit_params = params[1]        

    # EB Condition
    elif (isinstance(params[0], c.TargetStarParameters) and
            isinstance(params[1], c.SecondaryStarParameters)):

        # do something triple
        print('Checking parameters for EB system')

        # To be conservative, choose the smallest reasonable masses
        # and the largest possible radii
        # This will make the stars orbit closer and being larger
        mass1 = 0.1
        radius1_au = 5.0 * cts.Rsun / cts.au
        mass2 = 0.1
        radius2_au = 5.0 * cts.Rsun / cts.au

        # Define object containing orbital parameters
        orbit_params = params[2]

    # BTP condition
    elif (isinstance(params[0], c.BackgroundStarParameters) and
          isinstance(params[1], c.PlanetParameters)):
        # do something planety
        print('Checking parameters for BTP system')

        # Get stellar mass
        mass1 = params[0].mass
        # get stellar radius
        # Again, to be conservative, choose LARGE radius
        radius1_au = 10.0 * cts.Rsun / cts.au
        
        # Get planet mass and orbital distance
        mass2 = params[1].mass_mearth * cts.GMearth / cts.GMsun
        radius2_au = params[1].radius_rearth * cts.Rearth / cts.au

        orbit_params = params[1]

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
