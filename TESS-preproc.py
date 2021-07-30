#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 22:36:12 2021

@author: agus
"""

#import pandas as pd
import dask.dataframe as dd

#https://archive.stsci.edu/tess/tic_ctl.html

# dtypes converted from
# https://archive.stsci.edu/missions/tess/catalogs/tic_v81/tic_column_description.txt

def gen_out(filename_large):
    dtypes = {    
        'ID': 'float64',
        'version': 'str',
        'HIP': 'int32',
        'TYC': 'str',
        'UCAC': 'str',
        'TWOMASS': 'str',
        'SDSS': 'float64',
        'ALLWISE': 'str',
        'GAIA': 'str',
        'APASS': 'str',
        'KIC': 'int32',
        'objType': 'str',
        'typeSrc': 'str',
        'ra': 'float64',
        'dec': 'float64',
        'POSflag': 'str',
        'pmRA': 'float64',
        'e_pmRA': 'float64',
        'pmDEC': 'float64',
        'e_pmDEC': 'float64',
        'PMflag': 'str',
        'plx': 'float64',
        'e_plx': 'float64',
        'PARflag': 'str',
        'gallong': 'float64',
        'gallat': 'float64',
        'eclong': 'float64',
        'eclat': 'float64',
        'Bmag': 'float64',
        'e_Bmag': 'float64',
        'Vmag': 'float64',
        'e_Vmag': 'float64',
        'umag': 'float64',
        'e_umag': 'float64',
        'gmag': 'float64',
        'e_gmag': 'float64',
        'rmag': 'float64',
        'e_rmag': 'float64',
        'imag': 'float64',
        'e_imag': 'float64',
        'zmag': 'float64',
        'e_zmag': 'float64',
        'Jmag': 'float64',
        'e_Jmag': 'float64',
        'Hmag': 'float64',
        'e_Hmag': 'float64',
        'Kmag': 'float64',
        'e_Kmag': 'float64',
        'TWOMflag': 'str',
        'prox': 'float64',
        'w1mag': 'float64',
        'e_w1mag': 'float64',
        'w2mag': 'float64',
        'e_w2mag': 'float64',
        'w3mag': 'float64',
        'e_w3mag': 'float64',
        'w4mag': 'float64',
        'e_w4mag': 'float64',
        'GAIAmag': 'float64',
        'e_GAIAmag': 'float64',
        'Tmag': 'float64',
        'e_Tmag': 'float64',
        'TESSflag': 'str',
        'SPFlag': 'str',
        'Teff': 'float64',
        'e_Teff': 'float64',
        'logg': 'float64',
        'e_logg': 'float64',
        'MH': 'float64',
        'e_MH': 'float64',
        'rad': 'float64',
        'e_rad': 'float64',
        'mass': 'float64',
        'e_mass': 'float64',
        'rho': 'float64',
        'e_rho': 'float64',
        'lumclass': 'str',
        'lum': 'float64',
        'e_lum': 'float64',
        'd': 'float64',
        'e_d': 'float64',
        'ebv': 'float64',
        'e_ebv': 'float64',    
        'numcont': 'int32',
        'contratio': 'float64',
        'disposition': 'str',
        'duplicate_id': 'float64',
        'priority': 'float64',
        'eneg_EBV': 'float64',
        'epos_EBV': 'float64',
        'EBVflag': 'str',
        'eneg_Mass': 'float64',
        'epos_Mass': 'float64',
        'eneg_Rad': 'float64',
        'epos_Rad': 'float64',
        'eneg_rho': 'float64',
        'epos_rho': 'float64',
        'eneg_logg': 'float64',
        'epos_logg': 'float64',
        'eneg_lum': 'float64',
        'epos_lum': 'float64',
        'eneg_dist': 'float64',
        'epos_dist': 'float64',
        'distflag': 'str',
        'eneg_Teff': 'float64',
        'epos_Teff': 'float64',
        'TeffFlag': 'str',
        'gaiabp': 'float64',
        'e_gaiabp': 'float64',
        'gaiarp': 'float64',
        'e_gaiarp': 'float64',
        'gaiaqflag': 'int32',
        'starchareFlag': 'str',
        'VmagFlag': 'str',
        'BmagFlag': 'str',
        'splists': 'str',
        'e_RA': 'float64',
        'e_Dec': 'float64',
        'RA_orig': 'float64',
        'Dec_orig': 'float64',
        'e_RA_orig': 'float64',
        'e_Dec_orig': 'float64',
        'raddflag': 'int32',
        'wdflag': 'int32',
        'objID': 'float64'
    }
    
    new_columns = [name for name in dtypes.keys()]
  
    # al parecer hay que hacer una conversión un poco más especíca  
    data = dd.read_csv(filename_large+".csv", names=new_columns, dtype={'disposition': 'object', 'wdflag': 'float64', 'Teff': 'float64', 'MH': 'float64', 'logg': 'float64'})
    
    select = data[data["Vmag"]<13.5]
 
    #if we have a nan in Teff, logg o plx drop the line
    select_all = select[["ID", "Teff", "logg", "MH", "plx"]].dropna(subset=['Teff', 'logg', 'plx'])

    select_MH = select[["MH"]].dropna()
    
    # just checking
    print("Stars", select_all.head(10))
    print("Metalicity",select_MH.head(10))
        
    # it's possible to generate the output file in parallel
    select_all.to_csv(filename_large+"_ID_TEFF_LOGG_MH_PLX.csv",single_file=True)
    select_MH.to_csv(filename_large+"_ID_MH.csv",single_file=True)

    
    #https://medium.com/analytics-vidhya/optimized-ways-to-read-large-csvs-in-python-ab2b36a7914e
    #https://examples.dask.org/dataframes/01-data-access.html
    #https://pythondata.com/dask-large-csv-python/
    #https://stackoverflow.com/questions/40627980/row-wise-selection-based-on-multiple-conditions-in-dask
    
    #seleccionar rows y eso
    #https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html



# just in case we need it in the future

def merge_csv(in_directory, out_filename):
    with open(out_filename, 'w') as outfile:
        for in_filename in in_directory:          
            with open(in_filename, 'r') as infile:
                # if your csv files have headers then you might want to burn a line here with `next(infile)
                for line in infile:
     #               outfile.write(line + '\n')
                    print(line)
    
  
  
folder = "./data/"
#filename_large =  folder+'tic_dec30_00S__28_00S'    
filename_large =  folder+'tic_dec88_00S__86_00S'    	
#filename_large =  folder+'tic_dec74_00S__72_00S'    	
# filename_large =  folder+'tic_dec58_00S__56_00S'    	
#filename_large =  folder+'tic_dec66_00S__64_00S'    	

gen_out(filename_large)
#merge_csv(filename_large+"ID_TEFF_LOGG_MH.csv/", filename_large+"ID_TEFF_LOGG_MH-OUT.csv")      
    
    
    


    
    