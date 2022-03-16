#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:38:50 2021

@author: rodrigo
"""
import numpy as np
# import pickle

# Import relevant modules from PASTIS
from pastis import isochrones, limbdarkening, photometry
from pastis.extlib import SAMdict, EMdict
from pastis.paths import filterpath, zeromagfile

# Initialise if needed
if not hasattr(limbdarkening, 'LDCs'):
    limbdarkening.initialize_limbdarkening(['Johnson-R', 'TESS'])

if not hasattr(photometry, 'Filters'):
    photometry.initialize_phot(['Johnson-R', 'TESS'], zeromagfile,
                               filterpath,
                               AMmodel=SAMdict['BT-settl'])
    # photometry.initialize_phot_WD()
if not hasattr(isochrones, 'maxz'):
    isochrones.interpol_tracks(EMdict['Dartmouth'])
    isochrones.prepare_tracks_target(EMdict['Dartmouth'])

# import core as c
from pastisML_tess import draw as d
# import parameters as p
from pastisML_tess import simulation as s

size = 1000

teff = np.random.randn(size)*20 + 5777
feh = np.random.randn(size)*0.01
logg = np.random.randn(size)*0.01 + 4.4
    
params = np.array([teff, logg, feh]).reshape(3, -1, order='F')
len(params)


scenarios_to_test = ['PLANET', 'BEB', 'TRIPLE', 'EB', 'BTP', 'PIB']
results = {}

for SCENARIO in scenarios_to_test:
    print('#### {} #####'.format(SCENARIO))
    input_dict, flag = d.draw_parameters(params, SCENARIO)
    
    object_list, rej = s.build_objects(input_dict, np.sum(flag), True)
    
    #  Construct light curves
    f = []
    periods = []
    periods_triple = []
    for obj in object_list:
        # Get period from object list
        if SCENARIO.lower() in ['pla', 'planet', 'btp']:
            P = obj[0].planets[0].orbital_parameters.P
        elif SCENARIO.lower() in ['beb', 'eb']:
            P = obj[0].orbital_parameters.P
        elif SCENARIO.lower() == 'triple':
            P = obj[0].object2.orbital_parameters.P
        elif SCENARIO.lower() == 'pib':
            P = obj[0].object2.planets[0].orbital_parameters.P

        if SCENARIO.lower() in ['pib', 'triple']:
            Ptriple = obj[0].orbital_parameters.P
            periods_triple.append(Ptriple)

        periods.append(P)

        

        # sample according to value of period
        tess_cadence_min = 2.0
        n_points = int(np.ceil(P * 24 * 60 / tess_cadence_min))
        tt = np.linspace(0, 1, n_points)
        from pastis.models import PHOT
    
        try:
            lci = PHOT.PASTIS_PHOT(tt, 'TESS',
                                   True, 0.0, 1.0, 0.0, *obj)
        except:
            continue
    
        f.append([lci, P, n_points])
           
    results[SCENARIO] = [object_list, rej, f]
    print('{}: Pmax = {:.2f}'.format(SCENARIO, np.max(periods)))
    try:
        print('{}: Ptriple = {:.2f}'.format(SCENARIO, np.max(periods_triple)))
    except:
        pass