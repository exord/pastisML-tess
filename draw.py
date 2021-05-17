#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:23:07 2021

Module with functions to randomly draw parameters for given transit scenario 
FP or planetary.

@author: rodrigo
"""
import os
# import numpy as np
# import pandas as pd

import core as c
import parameters as p

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
    if scenario in ['PLA', 'planet']:
        pla_params = _draw_parameters_pla(ticstar, **kwargs)
        
        # Construct planet dict for pastis        
        # planetdict = pla_params[-1].to_pastis()
        # Add tree-like structure
        planetdict = {'star1': 'Target1', 'planet1': 'Planet1'}
        # planetdict['P'] = pla_params[0].period

        input_dict = {'Target1': ticstar.to_pastis(),
                      'Planet1': pla_params.to_pastis(),
                      'PlanSys1': planetdict}        
        
    elif scenario in ['BEB', 'beb']:
        beb_params = _draw_parameters_beb(ticstar, **kwargs)
        
        # Construct binary dict for pastis
        binarydict = beb_params[-1].to_pastis()
        # Add tree-like structure
        binarydict.update({'star1': 'Blend1', 'star2': 'Blend2'})
        binarydict['P'] = beb_params[1].period
        
        input_dict = {'Target1': ticstar.to_pastis(),
                      'Blend1': beb_params[0].to_pastis(),
                      'Blend2': beb_params[1].to_pastis(),
                      'IsoBinary1': binarydict
                      }
    
    return input_dict


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
    bkg_primary = c.BackgroundStarParameters()
    # Draw parameters for primary
    bkg_primary.draw(len(ticstar))
    
    # Build secondary
    bkg_secondary = c.SecondaryStarParameters(bkg_primary)
    bkg_secondary.draw()
    
    # Draw orbit
    orbit = c.OrbitParameters(orbittype='binary')
    orbit.draw(len(ticstar))
    
    return [bkg_primary, bkg_secondary, orbit]
    