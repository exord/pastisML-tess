#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:20:57 2021

@author: rodrigo
"""
import os
import pickle

scn = 'PLANET'

configpath = '/data/pastis/configfiles/BTPastis11'

if scn=='BEB':
    configfile = ('BTPastis11_PLANET_b0.00_Rpl0.1028_Mp1.00_SNR20_'
                  'scenario{}.pastis'.format(scn))
elif scn=='PLANET':
    configfile = ('BTPastis11_PLANET_b0.00_Rpl0.0716_Mp1.00_SNR50_'
                  'scenario{}.pastis'.format(scn))

f = open(os.path.join(configpath, configfile), 'rb')
input_dict = pickle.load(f, encoding='latin')[1]

print(input_dict.keys())