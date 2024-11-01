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

#from pastisML_tess import draw as d
import draw as d
import parameters as p

# Because pastis is crap, we can only import this after initialisation
#from pastisML_tess import simulation as s
import simulation as s


# Read parameters
from parameters import SCENARIO, NSIMU_PER_TIC_STAR, THETAMIN_DEG
#import SCENARIO, NSIMU_PER_TIC_STAR

#to force garbage collection
import gc


def gen_files(params, part_num, pd_tess, **kwargs):
    # Draw parameters for scenario
    
    input_dict, flag = d.draw_parameters(params, SCENARIO, 
                                         nsimu=NSIMU_PER_TIC_STAR,
                                         thetamin_deg=THETAMIN_DEG,
                                         **kwargs)
        
    # Create objects 
    object_list, rej = s.build_objects(input_dict, np.sum(flag), True)
    
    # Compute model light curves
    lc = s.lightcurves(object_list, scenario=SCENARIO, lc_cadence_min=2.0)
      
    out_file = open("./simulations/"+SCENARIO+"-lightcurves-index-"+str(part_num)+".txt", "w")

    out_file.write("Rejected: \n")
    out_file.write(str(rej) + "\n" )
    out_file.write("------------- \n")

    #periods candidate
    if SCENARIO=='BEB' or SCENARIO=='TRIPLE' :
        periods_dict = input_dict['IsoBinary1']['P']
    elif SCENARIO=='PLA' or SCENARIO=='BTP' or SCENARIO=='PIB':
        planet_key = input_dict['PlanSys1']['planet1']
        periods_dict = input_dict[planet_key]['P']
    elif SCENARIO=='EB':
        periods_dict =  input_dict['qBinary1']['P']

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

        #We search for the object used to create the LC

        obj = object_list[simu_number]
        
        if SCENARIO == 'BEB':
            out_file.write("star1_mact" + " "+ str(obj[0].star1.mact) + ",")
            out_file.write("star2_mact" + " "+ str(obj[0].star2.mact) + ",")
            out_file.write("star1_R" + " "+ str(obj[0].star1.R) + ",")
            out_file.write("star2_R" + " "+ str(obj[0].star2.R) + ",")            
            out_file.write("target_mact" + " "+ str(obj[1].mact) + ",")            
            out_file.write("target_R" + " "+ str(obj[1].R) + ",")                    
            out_file.write("target_L" + " "+ str(obj[1].L) + ",")                                
            
        if SCENARIO == 'PLA':
            out_file.write("star_mact" + " "+ str(obj[0].star.mact) + ",")
            out_file.write("star_R" + " "+ str(obj[0].star.R) + ",")
            out_file.write("star_L" + " "+ str(obj[0].star.L) + ",")
    
        if SCENARIO == 'EB':
            out_file.write("target_mact" + " "+ str(obj[0].star1.mact) + ",")
            out_file.write("star2_mact" + " "+ str(obj[0].star2.mact) + ",")
            out_file.write("target_R" + " "+ str(obj[0].star1.R) + ",")
            out_file.write("star2_R" + "  "+ str(obj[0].star2.R) + ",")
            out_file.write("target_L" + " "+ str(obj[0].star1.L) + ",")
            out_file.write("star2_L" + " "+ str(obj[0].star2.L) + ",")
    
        if SCENARIO == 'TRIPLE':
            out_file.write("star1_mact" + " "+ str(obj[0].object2.star1.mact) + ",")
            out_file.write("star2_mact" + " "+ str(obj[0].object2.star2.mact) + ",")
            out_file.write("star1_R" + " "+ str(obj[0].object2.star1.R) + ",")
            out_file.write("star2_R" + " "+ str(obj[0].object2.star2.R) + ",")
            out_file.write("star1_L" + " "+ str(obj[0].object2.star1.L) + ",")
            out_file.write("star2_L" + " "+ str(obj[0].object2.star2.L) + ",")                        
            out_file.write("target_mact" + " "+ str(obj[0].object1.mact) + ",")            
            out_file.write("target_R" + " "+ str(obj[0].object1.R) + ",")                    
            out_file.write("target_L" + " "+ str(obj[0].object1.L) + ",")         
            
        if SCENARIO == 'BTP':
            out_file.write("star_mact" + " "+ str(obj[0].star.mact) + ",")
            out_file.write("star_R" + " "+ str(obj[0].star.R) + ",")
            out_file.write("star_L" + " "+ str(obj[0].star.L) + ",")
            out_file.write("target_mact" + " "+ str(obj[1].mact) + ",")            
            out_file.write("target_R" + " "+ str(obj[1].R) + ",")                    
            out_file.write("target_L" + " "+ str(obj[1].L) + ",")               
            
        if SCENARIO == 'PIB':
            out_file.write("star_mact" + " "+ str(obj[0].object2.star.mact) + ",")
            out_file.write("star_R" + " "+ str(obj[0].object2.star.R) + ",")
            out_file.write("star_L" + " "+ str(obj[0].object2.star.L) + ",")
            out_file.write("target_mact" + " "+ str(obj[0].object1.mact) + ",")            
            out_file.write("target_R" + " "+ str(obj[0].object1.R) + ",")                    
            out_file.write("target_L" + " "+ str(obj[0].object1.L) + ",")                  
                
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
#just the names, next version just parse some directory or something
filenames = ["tic_dec66_00S__64_00S_","tic_dec58_00S__56_00S_","tic_dec30_00S__28_00S_","tic_dec74_00S__72_00S_","tic_dec62_00S__60_00S_","tic_dec28_00S__26_00S_","tic_dec88_00S__86_00S_"]

filenames = filenames * 5 #quick and dirty way to repeat stars, I love it

full_data=[]
full_data_PD=pd.DataFrame([])

for file in filenames:
    TEFF_LOGG_MH_data_file = file+"ID_TEFF_LOGG_MH.csv"
    MH_data_file = file+"ID_MH.csv"
    
    print("Reading:",TEFF_LOGG_MH_data_file)

    #read files   
    data_pd = pd.read_csv(TEFF_LOGG_MH_data_file)
    #we need the pandas 
    data = data_pd[['Teff','logg','MH']].values.tolist()

    MH_data_pd = pd.read_csv(MH_data_file)
    MH_data = MH_data_pd['MH'].values.tolist()

    #debe haber una forma mas numpy para esto    
    #filling the MH data
    for star in data:
        if np.isnan(star[2]):
            star[2] = np.random.choice(MH_data)

    full_data = full_data+data
    full_data_PD = pd.concat([full_data_PD, data_pd])

#Just to split into batchs 
start = 0
full_data = np.asarray(full_data)
for part, end in enumerate(np.linspace(20000, len(full_data), 16, dtype=int)):
    if part>=0: #to avoid restart in case of failure
        print (start, end, "Part:", part)
        TEFF_LOGG_MH_slice = full_data[start:end]
        #para usar el mismo formato que habia antes
        params = TEFF_LOGG_MH_slice.flatten().reshape(3, len(TEFF_LOGG_MH_slice), order='F')
        gen_files(params, part, full_data_PD, method='uniform')
    start = end
    #nunca se si esto funca o no, just in case
    gc.collect()