#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:28:29 2021

@author: rodrigo
"""
import numpy as np
import pandas as pd

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
import parameters as p

# Because pastis is crap, we can only import this after initialisation
from pastisML_tess import simulation as s

# Read parameters
from parameters import SCENARIO, NSIMU_PER_TIC_STAR

#to force garbage collection
import gc


def gen_files(params, part_num, pd_tess):
    # Draw parameters for scenario
    
    input_dict, flag = d.draw_parameters(params, SCENARIO, 
                                         nsimu=NSIMU_PER_TIC_STAR)
        
    # Create objects 
    object_list, rej = s.build_objects(input_dict, np.sum(flag), True)
    
    # Compute model light curves
    lc = s.lightcurves(object_list, scenario=SCENARIO, lc_cadence_min=2.0)
    
    # TODO
    # Conservar y entregar rej, el diccionario que contiene cuántos sistemas
    # se cortaron por cada uno de los criterios.

    for obj in object_list:
        if SCENARIO == 'BEB':
            # de todas las estrellas: mact, R, L
            obj[0].star1.mact
            obj[0].star2.mact
            obj[0].star1.R
            obj[0].star2.R
            
            obj[1].target.mact
            obj[1].target.R
            obj[1].target.L
            
        if SCENARIO == 'PLA':
            obj[0].star.mact
            obj[0].star.R
            obj[0].star.L
            
            
    # Save index and simulations
    
    out_file = open("./simulations/"+SCENARIO+"-lightcurves-index-"+str(part_num)+".txt", "w")
    
    #periods candidate
    if SCENARIO=='BEB':
        periods_dict = input_dict['IsoBinary1']['P']
    elif SCENARIO=='PLA':
        planet_key = input_dict['PlanSys1']['planet1']
        periods_dict = input_dict[planet_key]['P']


    for simu_number in range(len(lc)):
        out_file_line=[]
      
        #which P was successfull    
        #TODO hay una forma de hacer mejor esto? es horrible
        pos_elem = np.where(periods_dict==lc[simu_number][1])[0][0]
        
        for obj in input_dict:
            if obj == 'Target1':
                teff_obj= input_dict[obj]['teff'][pos_elem]
                logg_obj= input_dict[obj]['logg'][pos_elem]           
                #aprendiendo pandas a los golpes :P
                id_obj = pd_tess[(pd_tess['Teff'] == teff_obj) & (pd_tess['logg'] == logg_obj)]['ID'].head(1).to_numpy()[0]
                out_file_line.append(("ID",id_obj))

            pd = input_dict[obj]
            for par in pd:
                if isinstance(pd[par], (np.ndarray, np.generic) ): 
                    out_file_line.append((par,pd[par][pos_elem]))
                else: #e.g. istar1:Blend1, 'star1': 'Target1', planet1': 'Planet1'
                    out_file_line.append((par,pd[par]))
                    
        #save simulation and values            
        simu_name = './simulations/'+SCENARIO+'-simu-'+str(part_num)+"-"+str(simu_number)+'.csv'   
        print("Saving slice:",part_num, "simulation:", simu_number)
        np.savetxt(simu_name, lc[simu_number][0], delimiter=',') #as np array
    
        for tuple in out_file_line:
                    out_file.write(str(tuple[0]) + " "+ str(tuple[1]) + ",")
                
        out_file.write(simu_name + "\n")
    out_file.close()
    #just in case, we force the garbage collection
    del lc
    del input_dict
    del object_list
    gc.collect()
    print("Done!")



# Read TIC star parameter list

## without real parameters
# teff = np.random.randn(NSIMU_PER_TIC_STAR)*20 + 5777
# feh = np.random.randn(NSIMU_PER_TIC_STAR)*0.01
# logg = np.random.randn(NSIMU_PER_TIC_STAR)*0.01 + 4.4


print("Reading input files")
#tess_ID_TEFF_LOGG_MH_filename_0 = "tic_dec30_00S__28_00S_ID_TEFF_LOGG_MH.csv"
#tess_MH_filename_0 = "tic_dec30_00S__28_00S_ID_MH.csv"


# just a quick and dirty fix to append two .csv

tess_ID_TEFF_LOGG_MH_filename_0 = "tic_dec66_00S__64_00S_ID_TEFF_LOGG_MH.csv"
tess_MH_filename_0 = "tic_dec66_00S__64_00S_ID_MH.csv"


pd_ID_TEFF_LOGG_MH_0 = pd.read_csv(tess_ID_TEFF_LOGG_MH_filename_0 )
TEFF_LOGG_MH_0 = pd_ID_TEFF_LOGG_MH_0[['Teff','logg','MH']].to_numpy()

MH_0 = pd.read_csv(tess_MH_filename_0)
MH_0 = MH_0['MH'].to_numpy()

#debe haber una forma mas numpy para esto
for a in TEFF_LOGG_MH_0:
    if np.isnan(a[2]):
        a[2] = np.random.choice(MH_0)


tess_ID_TEFF_LOGG_MH_filename_1 = "tic_dec58_00S__56_00S_ID_TEFF_LOGG_MH.csv"
tess_MH_filename_1 = "tic_dec58_00S__56_00S_ID_MH.csv"

pd_ID_TEFF_LOGG_MH_1 = pd.read_csv(tess_ID_TEFF_LOGG_MH_filename_1 )
TEFF_LOGG_MH_1 = pd_ID_TEFF_LOGG_MH_1[['Teff','logg','MH']].to_numpy()

MH_1 = pd.read_csv(tess_MH_filename_1)
MH_1 = MH_1['MH'].to_numpy()

#debe haber una forma mas numpy para esto
for a in TEFF_LOGG_MH_1:
    if np.isnan(a[2]):
        a[2] = np.random.choice(MH_1)


TEFF_LOGG_MH = np.concatenate((TEFF_LOGG_MH_0, TEFF_LOGG_MH_1))
frames = [pd_ID_TEFF_LOGG_MH_0, pd_ID_TEFF_LOGG_MH_1]
pd_ID_TEFF_LOGG_MH = pd.concat(frames)

#para partir en batch de masomenos 10k estrellas, enumero las particiones también
start = 0
for part, end in enumerate(np.linspace(10000, len(TEFF_LOGG_MH), 16, dtype=int)):
    if part>=0: #para cuando se cortaba 
        print (start, end, "Part:", part)
        TEFF_LOGG_MH_slice = TEFF_LOGG_MH[start:end]
        #para usar el mismo formato que habia antes
        params = TEFF_LOGG_MH_slice.flatten().reshape(3, len(TEFF_LOGG_MH_slice), order='F')
        gen_files(params, part, pd_ID_TEFF_LOGG_MH)
        # break
    start = end
    gc.collect()



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
