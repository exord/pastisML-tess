#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:23:07 2021

Module with functions to randomly draw parameters for given transit scenario 
FP or planetary.

@author: rodrigo
"""
import os
import numpy as np
import pandas as pd

import core as c

homedir = os.getenv('HOME')
mldir = os.path.join(homedir, 'EXOML', 'TESS-pastis')

def draw_parameters(params, scenario, nsimu=1):
    """
    Draw the parameters for a set of target (TIC) star.
    
    The draw is conditional on the parameter sets, representing the TIC star.
    
    :param iterable params: shape (nparams, size), list of parameters of TIC 
    star. Is the second dimension is > 1, then nsimu draws are performed for 
    each parameter set.
    
    :param str scenario: defines the scenario for which to draw parameters
    
    :param int nsimu: size of parameter set to draw
    """
    ticstar = c.TargetStarParameters(params)
    ticstar.draw()
    
    if scenario in ['PLA', 'planet']:
        _draw_parameters_pla(ticstar)
        
    elif scenario in ['BEB', 'beb']:
        beb_params = _draw_parameters_beb(ticstar)
        
        binarydict = beb_params[-1].to_pastis()
        binarydict.update({'star1': 'Blend1', 'star2': 'Blend2'})
        binarydict['P'] = beb_params[1].period
        
        input_dict = {'Target1': ticstar.to_pastis(),
                      'Blend1': beb_params[0].to_pastis(),
                      'Blend2': beb_params[1].to_pastis(),
                      'IsoBinary1': binarydict
                      }
    
    return input_dict


def _draw_parameters_pla(ticstar):
    
    # Parameters represent Teff, logg and Fe/H of planet host
    teff, logg, feh = params
    
    # LDC coefficients will be drawn using pastis
    

def _draw_parameters_beb(ticstar):
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
    orbit = c.OrbitParameters()
    orbit.draw(len(ticstar))
    
    return [bkg_primary, bkg_secondary, orbit]
    