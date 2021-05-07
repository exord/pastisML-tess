#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:28:29 2021

@author: rodrigo
"""
import numpy as np

# Import necessary pastis modules and variables
from pastis import isochrones, limbdarkening, photometry
from pastis import extlib, paths

# Import internal modules
import draw as d

# Read parameters
from parameters import SCENARIO, NSIMU_PER_TIC_STAR

# Read / Create TIC star parameter list
teff = np.random.randn(NSIMU_PER_TIC_STAR)*20 + 5777
feh = np.random.randn(NSIMU_PER_TIC_STAR)*0.01
logg = np.random.randn(NSIMU_PER_TIC_STAR)*0.01 + 4.4

nrepeat = 20
params = np.array([teff, logg, feh]*nrepeat).reshape(3, nrepeat, order='F')


# Initialise pastis

# Limb darkening coefficients
limbdarkening.initialize_limbdarkening(['TESS', ])

# Photometric bands
photometry.initialize_phot(['TESS', ], paths.zeromagfile, paths.filterpath, 
                           AMmodel=extlib.SAMdict['BT-settl'])

# Stellar tracks and isochrones
isochrones.interpol_tracks(extlib.EMdict['Dartmouth'])
isochrones.prepare_tracks_target(extlib.EMdict['Dartmouth'])

# Because pastis is crap, we can only import this after initialisation
import simulation as s

# Draw parameters for scenario
input_dict = d.draw_parameters(params, SCENARIO, nsimu=NSIMU_PER_TIC_STAR)

# Create objects 
object_list = s.build_objects(input_dict, len(params.T))

# Compute model light curves
lc = s.lightcurves(object_list, scenario=SCENARIO, lc_cadence_min=2.0)

# TODO: Save results