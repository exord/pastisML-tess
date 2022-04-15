#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:53:49 2021

@author: rodrigo
"""
import numpy as np

from pastis import ObjectBuilder as ob
from pastis import AstroClasses as ac
from pastis.models import PHOT
from pastis.exceptions import EvolTrackError, EBOPparamError
import pickle

'''
from . import constants as cts
from . import utils as u
from . import parameters as p
'''
import constants as cts
import utils as u
import parameters as p


def build_objects(input_dict, nsimu, return_rejected_stats, verbose=False, **kwargs):
    """
    Build pastis objects.

    Uses parameters in input_dict, where more than one sample can in
    principle be provided.

    Parameters
    ----------
    :param int nsimu: number of simulations to perform
    :param bool return_rejected_stats: whether to return dict with number of rejected systems per cause
    :param bool verbose: verbose flag

    Other params
    ------------
    :param float min_depth: minimum depth filter in ppm (overrides default in parameters module)
    :param float max_mag_diff: maximum magnitude difference between blended system and foreground bright star (overried default in parameters module)
    """
    min_depth = kwargs.pop('min_depth', p.MIN_DEPTH)
    max_mag_diff = kwargs.pop('max_mag_diff', p.MAX_MAG_DIFF)

    # Define dictionary that will be passed to PASTIS object builder
    dd = {}
    # Intialise list with objects (use list instead of array
    # because constructions may fail)
    objs = []

    rejected = {'inclination': 0,
                'brightness': 0,
                'isochrone': 0,
                'depth': 0,
                'ebop': 0}

    # Iterate over number of simulations
    for i in range(nsimu):

        if i % 100 == 0:
            print(i)
        # Iterate over objects
        for obj in input_dict:
            dd[obj] = {}

            # Iterate over parameters
            for par in input_dict[obj]:
                # Change name of problematic parameters
                if par == 'ph_tr':
                    par2 = 'T0'
                else:
                    par2 = par

                # Read parameter value (it may eventually be an array)
                x = input_dict[obj][par]

                # If an array, keep element i
                if isinstance(x, np.ndarray):
                    # For pastis compatibility, this must be a list
                    dd[obj][par2] = [input_dict[obj][par][i], ]

                # If a float, check that only one simulation is requested
                elif isinstance(x, float):
                    assert nsimu == 1, ('More than one simulation requested '
                                        'but input_dict has only one value '
                                        'for parameter {} of object {}'
                                        ''.format(par, obj))

                    dd[obj][par2] = [input_dict[obj][par], ]

                # If string, it must be an indicator parameter, such as the
                # name of the components in a binary. Carry on...
                elif isinstance(x, str):
                    dd[obj][par2] = input_dict[obj][par]
                else:
                    raise TypeError('Could not handle whatever is in'
                                    ' parameter {} of object {}'
                                    ''.format(par, obj))

        # Construct objects with full parameter
        try:
            system = ob.ObjectBuilder(dd)
        except EvolTrackError as ex:
            # If fail in Evolution track interpolation, print error and
            # continue
            if verbose: print(ex)
            rejected['isochrone'] += 1
            continue

        # Check again for transits, this time using realistic parameters
        if not check_eclipses(system):
            if verbose: print('Not transiting')
            rejected['inclination'] += 1
            continue

        # Check system brightness
        if not  check_brightness(system, max_mag_diff):
            if verbose: print('Magnitud difference > {}'.format(max_mag_diff))
            rejected['brightness'] += 1
            continue

        # Check depth
        try:
            if not check_depth(system, min_depth):
                if verbose: print('Eclipse / transit depth < {}'.format(min_depth))
                rejected['depth'] += 1
                continue
            else:
                pass
            
        except (EBOPparamError, AssertionError):
            if verbose: print('Encoutered EBOP limit when testing for depth or NaNs')
            rejected['ebop'] += 1
            continue

        objs.append(system)

    if return_rejected_stats:
        return objs, rejected
    else:
        return objs


def check_eclipses(objects):
    """Verify if a given system actually eclipses / transits."""
    
    # Introspection

    # SINGLE OBJECT (UNDILUTED / BOUND) SCENARIOS
    if len(objects) == 1:
        
        if isinstance(objects[0], ac.PlanSys):
            oo = objects[0]
            primary = oo.star
            secondary = oo.planets[0]
            orbit_params = secondary.orbital_parameters

        # EB system
        elif isinstance(objects[0], ac.qBinary):
            oo = objects[0]
            primary = oo.star1
            secondary = oo.star2
            orbit_params = oo.orbital_parameters

        # TRIPLE / PIB CASES   
        elif isinstance(objects[0], ac.Triple):
        
            # Check eclipse of bounded binary (by convention this is object2)
            oo = objects[0].object2
        
            # Hierarchichal triple
            if isinstance(oo, ac.IsoBinary) or isinstance(oo, ac.qBinary):
                primary = oo.star1
                secondary = oo.star2
                orbit_params = oo.orbital_parameters
            # PIB
            elif isinstance(oo, ac.PlanSys):
                primary = oo.star
                secondary = oo.planets[0]
                orbit_params = secondary.orbital_parameters

    # DOUBLE / DILUTED SCENARIOS
    elif len(objects) == 2:

        if np.any([isinstance(oo, ac.IsoBinary) for oo in objects]):

            eb = np.array([isinstance(oo, ac.IsoBinary) for oo in objects])
            oo = np.array(objects)[eb][0]

            primary = oo.star1
            secondary = oo.star2
            orbit_params = oo.orbital_parameters

        elif np.any([isinstance(oo, ac.PlanSys) for oo in objects]):

            plansys = np.array([isinstance(oo, ac.PlanSys) for oo in objects])
            oo = np.array(objects)[plansys][0]

            primary = oo.star
            secondary = oo.planets[0]
            orbit_params = secondary.orbital_parameters

    # Get masses and radii
    mass1 = primary.mact
    radius1_au = primary.R * cts.Rsun / cts.au

    mass2 = secondary.mact
    radius2_au = secondary.R * cts.Rsun / cts.au

    # Get relevant orbital parameters
    periods = orbit_params.P
    ecc = orbit_params.ecc
    omega_deg = orbit_params.omega * 180 / np.pi
    incl_rad = orbit_params.incl

    # Compute separation at inferior conjunction
    sma_au = u.sma(periods, mass1, mass2)
    r0 = u.r_infconj(ecc, omega_deg, sma_au / radius1_au)

    # compute impact parameter
    b = r0 * np.cos(incl_rad)

    orbit_params.b = b
    orbit_params.r0 = r0

    # Return condition of transit
    return b <= (1 + radius2_au/radius1_au)


def check_brightness(objects, max_mag_diff=None):
    """Verify if diluted system is bright enough to produce FP."""
    # Get max mag diff
    if max_mag_diff is None:
        mmd = p.MAX_MAG_DIFF
    else:
        mmd = max_mag_diff

    # Introspection
    # Case PLA, EB or Triple/PiB (i.e. single-object scenarios)
    if len(objects) == 1:
        
        # Planet or EB scenarios
        if isinstance(objects[0], ac.PlanSys) or isinstance(objects[0], ac.qBinary):
            return True

        elif isinstance(objects[0], ac.Triple):
            # Check brightness of bounded binary or
            # planetary system (object2) with respect to target star
            # (by convention this is object1)
            target = objects[0].object1
            oo = objects[0].object2

            # Triple scenario
            if isinstance(oo, ac.IsoBinary) or isinstance(oo, ac.qBinary):
                primary = oo.star1
            
            # PiB scenario
            elif isinstance(oo, ac.PlanSys):
                primary = oo.star

    # DOUBLE / DILUTED SCENARIOS
    elif len(objects) == 2:

        target = getTarget(objects)

        # BEB scenario
        if np.any([isinstance(oo, ac.IsoBinary) for oo in objects]):
            primary = getEB(objects).star1
        
        # BTP scenario
        elif np.any([isinstance(oo, ac.PlanSys) for oo in objects]):
            primary = getPlanSys(objects).star
        
    # compute delta magnitude
    delta_mag = primary.get_mag('TESS') - target.get_mag('TESS')
    return (delta_mag > 0) and (delta_mag < mmd)
        
    
def check_depth(objects, min_depth=None):
    """
    Check that system has eclipses / transits deeper than min_depth.
    
    :param float min_depth: minimum depth on part-per-millon. If None take 
    value from params 
    """
    # Get max mag diff
    if min_depth is None:
        md = p.MIN_DEPTH
    else:
        md = min_depth

    # Evaluate curve at phase 0 (also evaluate at 0.5 to avoid 
    # PASTIS interpolation; long story...)
    lci = PHOT.PASTIS_PHOT(np.array([0.0, 0.5]), 'TESS', True, 0.0, 1.0, 0.0, 
                           *objects)        

    return (1 - lci.flatten()[0]) * 1e6 > md


def getEB(objects):
    """Return the element of the objects list that is an IsoBinary."""
    eb_ = np.array([isinstance(oo, ac.IsoBinary) for oo in objects])
    return np.array(objects)[eb_][0]


def getTarget(objects):
    """Return the element of the objects list that is a Target star."""
    target_ = np.array([isinstance(oo, ac.Target) for oo in objects])
    return np.array(objects)[target_][0]


def getPlanSys(objects):
    """Return the element of the objects list that is a Target star."""
    plansys_ = np.array([isinstance(oo, ac.PlanSys) for oo in objects])
    return np.array(objects)[plansys_][0]


def lightcurves(object_list, scenario='PLA', lc_cadence_min=2.0):
    """
    Build PASTIS light curves.

    For each element in the input list, run the PASTIS photometric model.

    :param iterable object_list: each element is a list of objects to
    pass to pastis.
    """
    #  Construct light curves
    f = []
    for obj in object_list:
        # Get period from object list
        if scenario.lower() in ['pla', 'planet', 'btp']:
            P = obj[0].planets[0].orbital_parameters.P
        elif scenario.lower() in ['beb', 'eb']:
            P = obj[0].orbital_parameters.P
        elif scenario.lower() == 'triple':
            P = obj[0].object2.orbital_parameters.P
        elif scenario.lower() == 'pib':
            P = obj[0].object2.planets[0].orbital_parameters.P

        # define number of points according to period and cadence
        n_points = np.int(np.ceil(P * 24 * 60 / lc_cadence_min))
        tt = np.linspace(0, 1, n_points)
        try:
            pickle.dump(obj[0], open("obj.p", "wb"))
#            print("LOGG:", objs[0].star.logg)
#            print("TEFF:", objs[0].star.teff)
            lci = PHOT.PASTIS_PHOT(tt, 'TESS',
                                   True, 0.0, 1.0, 0.0, *obj)
        except EBOPparamError as ex:
            print(ex)
            continue
        except AssertionError as ex:
            print(ex)
            continue

        # append light curve, period and number of points
        f.append([lci, P, n_points])
    return f
