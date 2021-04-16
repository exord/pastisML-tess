#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:38:50 2021

@author: rodrigo
"""
import numpy as np
import pickle

import core as c
import draw as d
size = 1

teff = np.random.randn(size)*20 + 5777
feh = np.random.randn(size)*0.01
logg = np.random.randn(size)*0.01 + 4.4

params = np.array([teff, logg, feh]*100).reshape(3, 100, order='F')

# tt = c.TargetStarParameters(params)

tt = c.BackgroundStarParameters()
tt.draw(size)
tt2 = c.SecondaryStarParameters(tt)
tt2.draw()
orbit = c.OrbitParameters()
orbit.draw(size)

# print(len(tt))

def test_full_draw(nsimu=1):
    return d.draw_parameters(params, 'BEB', nsimu=nsimu)


def test_object_build():
    from pastis import ObjectBuilder as ob

    input_dict = test_full_draw()

    return ob.ObjectBuilder(input_dict)


if __name__ == '__main__':
    from pastis import isochrones, limbdarkening, photometry
    from pastis.extlib import SAMdict, EMdict
    from pastis.paths import filterpath, zeromagfile

    limbdarkening.initialize_limbdarkening(['Johnson-R',])

    photometry.initialize_phot(['Johnson-R',], zeromagfile, filterpath,
                                AMmodel=SAMdict['BT-settl'])
    #photometry.initialize_phot_WD()

    isochrones.interpol_tracks(EMdict['Dartmouth'])
    isochrones.prepare_tracks_target(EMdict['Dartmouth'])

    # input_dict = test_full_draw()
    
    # Read input dict constructed previously
    with open('/Users/rodrigo/code/python/packages/'
              'pastisML-tess-new/input_dict_beb.dat', 
              'rb') as f:
        # pickle.dump(input_dict, f)
        input_dict = pickle.load(f)

    objs = []

    # dd = input_dict.copy()
    from pastis import ObjectBuilder as ob
    dd = {}
    for i in range(params.shape[-1]):
        print(i)
        for obj in input_dict:
            dd[obj] = {}
            for par in input_dict[obj]:
                if par == 'ph_tr':
                    par2 = 'T0'
                else:
                    par2 = par
                # print(obj, par)
                x = input_dict[obj][par]
                if isinstance(x, np.ndarray):
                    # For pastis compatibility, this must be a list
                    dd[obj][par2] = [input_dict[obj][par][i],]
                elif isinstance(x, str):
                    dd[obj][par2] = input_dict[obj][par]
                else:
                    raise TypeError('WTF?')
                    

        from pastis.exceptions import EvolTrackError
        try:
            objs.append(ob.ObjectBuilder(dd))
        except EvolTrackError as ex:
            print(ex)
            continue
