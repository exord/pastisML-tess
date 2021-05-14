#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:28:29 2021

@author: rodrigo
"""
import numpy as np
import pandas as pd

# Import necessary pastis modules and variables
from pastis import isochrones, limbdarkening, photometry
from pastis import extlib, paths

# Import internal modules
import draw as d

# Read parameters
from parameters import SCENARIO, NSIMU_PER_TIC_STAR

# Read / Create TIC star parameter list
## without real parameters
# teff = np.random.randn(NSIMU_PER_TIC_STAR)*20 + 5777
# feh = np.random.randn(NSIMU_PER_TIC_STAR)*0.01
# logg = np.random.randn(NSIMU_PER_TIC_STAR)*0.01 + 4.4

# nrepeat = 20
# params = np.array([teff, logg, feh]*nrepeat).reshape(3, nrepeat, order='F')

print("Reading input files")
tess_TEFF_LOGG_MH_filename = "tic_dec30_00S__28_00S_ID_TEFF_LOGG_MH.csv"
tess_MH_filename = "tic_dec30_00S__28_00S_ID_MH.csv"

TEFF_LOGG_MH = pd.read_csv(tess_TEFF_LOGG_MH_filename)
TEFF_LOGG_MH = TEFF_LOGG_MH[['Teff','logg','MH']].to_numpy()

MH = pd.read_csv(tess_MH_filename)
MH = MH['MH'].to_numpy()

#debe haber una forma mas numpy para esto
for a in TEFF_LOGG_MH:
    if np.isnan(a[2]):
        a[2] = np.random.choice(MH)

#para poder usar el mismo formato que habia antes
params = TEFF_LOGG_MH.flatten().reshape(3, len(TEFF_LOGG_MH), order='F')

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

# Save index and simulations

out_file = open("./simulations/lightcurves-index.txt", "w")

#periods candidate
if SCENARIO=='BEB':
    periods_dict = input_dict['IsoBinary1']['P']
#if SCENARIO=='PLA':

for simu_number in range(len(lc)):
    out_file_line=[]
  
    #which P was successfull    
    #TODO hay una forma de hacer mejor esto? es horrible
    pos_elem = np.where(periods_dict==lc[simu_number][1])[0][0]
    
    for obj in input_dict:
        pd = input_dict[obj]
        for par in pd:
            if isinstance(pd[par], (np.ndarray, np.generic) ): 
                out_file_line.append((par,pd[par][pos_elem]))
            else: #e.g. istar1:Blend1
                out_file_line.append((par,pd[par]))
                
    #save simulation and values            
    simu_name = './simulations/simu-'+str(simu_number)+'.csv'   
    np.savetxt(simu_name, lc[simu_number][0], delimiter=',') #as np array

    for tuple in out_file_line:
                out_file.write(str(tuple[0]) + " "+ str(tuple[1]) + ",")
            
    out_file.write(simu_name + "\n")
out_file.close()

'''

ipdb> input_dict['IsoBinary1']['P']
array([ 462.45053955,   78.89342325,  338.3836427 , 3258.96699415,
         82.21294071,  552.87109946,   22.42308718,  287.97043385,
         15.41934606,  744.23270813, 6681.42561902,  248.10438678,
         15.08143364,  855.28509171,   70.18562019,   13.73334664,
        366.45847238,  434.75007145,  264.01064468,   10.38201648])

ipdb> lc[0][1]
287.9704338469746

ipdb> lc[1][1]
744.2327081324081

ipdb> lc[2][1]
6681.425619021072
'''

