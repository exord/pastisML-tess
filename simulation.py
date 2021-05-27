#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:53:49 2021

@author: rodrigo
"""
import numpy as np

from pastis import ObjectBuilder as ob
from pastis.models import PHOT
from pastis.exceptions import EvolTrackError, EBOPparamError
import pickle

def build_objects(input_dict, nsimu):
    """
    Build pastis objects.
    
    Uses parameters in input_dict, where more than one sample can in 
    principle be provided.
    """
    # Define dictionary that will be passed to PASTIS object builder
    dd = {}
    # Intialise list with objects (use list instead of array 
    # because constructions may fail)
    objs = []
    # Iterate over number of simulations
    for i in range(nsimu):
        
        if i%100 == 0:
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
                    dd[obj][par2] = [input_dict[obj][par][i],]
                
                # If a float, check that only one simulation is requested
                elif isinstance(x, float):
                    assert nsimu == 1, ('More than one simulation requested '
                                        'but input_dict has only one value '
                                        'for parameter {} of object {}'
                                        ''.format(par, obj))
                    
                    dd[obj][par2] = [input_dict[obj][par],]
                
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
            objs.append(ob.ObjectBuilder(dd))
        except EvolTrackError as ex:
            # If fail in Evolution track interpolation, print error and 
            # continue
            print(ex)
            continue

    return objs
    

def lightcurves(object_list, scenario='PLA', lc_cadence_min=2.0):
    """
    Build PASTIS light curves.
    
    For each element in the input list, run the PASTIS photometric model.
    
    :param iterable object_list: each element is a list of objects to 
    pass to pastis.
    """
    #  Construct light curves
    f = []
    for objs in object_list:
        # Get period from object list
        if scenario == 'PLA':
            P = objs[0].planets[0].orbital_parameters.P
        elif scenario == 'BEB':
            P = objs[0].orbital_parameters.P
                    
        # define number of points according to period and cadence
        n_points = np.int(np.ceil(P * 24 * 60 / lc_cadence_min))
        tt = np.linspace(0, 1, n_points)
        try:
            pickle.dump( objs[0], open( "obj.p", "wb" ) )
#            print("LOGG:", objs[0].star.logg)
#            print("TEFF:", objs[0].star.teff)            
            lci = PHOT.PASTIS_PHOT(tt, 'TESS', 
                                   True, 0.0, 1.0, 0.0, *objs)
        except EBOPparamError as ex:
            print(ex)
            continue
        # append light curve, period and number of points
        f.append([lci, P, n_points])
    return f