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
from . import draw as d
# import parameters as p
from . import simulation as s

size = 10000

teff = np.random.randn(size)*20 + 5777
feh = np.random.randn(size)*0.01
logg = np.random.randn(size)*0.01 + 4.4

params = np.array([teff, logg, feh]).reshape(3, -1, order='F')
len(params)

SCENARIO = 'BEB'

input_dict = d.draw_parameters(params, SCENARIO)
object_list, rej = s.build_objects(input_dict, 3000, True)

#  Construct light curves
f = []
for obj in object_list:
    # Get period from object list
    if SCENARIO == 'PLA':
        P = obj[0].planets[0].orbital_parameters.P
    elif SCENARIO == 'BEB':
        P = obj[0].orbital_parameters.P

    # sample according to value of period
    tess_cadence_min = 2.0
    n_points = np.int(np.ceil(P * 24 * 60 / tess_cadence_min))
    tt = np.linspace(0, 1, n_points)
    from pastis.models import PHOT

    try:
        lci = PHOT.PASTIS_PHOT(tt, 'TESS',
                               True, 0.0, 1.0, 0.0, *obj)
    except:
        continue

    f.append([lci, P, n_points])


# =============================================================================
# if __name__ == '__main__':
#
#     scenario = 'PLA'
#
#     # Initialise pastis
#     from pastis import isochrones, limbdarkening, photometry
#     from pastis.extlib import SAMdict, EMdict
#     from pastis.paths import filterpath, zeromagfile
#
#     limbdarkening.initialize_limbdarkening(['Johnson-R', 'TESS'])
#
#     photometry.initialize_phot(['Johnson-R', 'TESS'], zeromagfile,
#                                  filterpath,
#                                 AMmodel=SAMdict['BT-settl'])
#     #photometry.initialize_phot_WD()
#
#     isochrones.interpol_tracks(EMdict['Dartmouth'])
#     isochrones.prepare_tracks_target(EMdict['Dartmouth'])
#
#     # Draw parameters for BEB
#     input_dict = d.draw_parameters(params, scenario, nsimu=1)
#
#     # Read input dict constructed previously
# # ==========================================================================
# #     with open('/Users/rodrigo/code/python/packages/'
# #               'pastisML-tess-new/input_dict_beb.dat',
# #               'rb') as f:
# #         # pickle.dump(input_dict, f)
# #         input_dict = pickle.load(f)
# # ==========================================================================
#
#     # Create pastis objects
#     objs = []
#
#     # dd = input_dict.copy()
#     from pastis import ObjectBuilder as ob
#
#     # Discitonary that will finally be passed to pastis
#     dd = {}
#     for i in range(params.shape[-1]):
#         print(i)
#         for obj in input_dict:
#             dd[obj] = {}
#             for par in input_dict[obj]:
#                 if par == 'ph_tr':
#                     par2 = 'T0'
#                 else:
#                     par2 = par
#                 # print(obj, par)
#                 x = input_dict[obj][par]
#                 if isinstance(x, np.ndarray):
#                     # For pastis compatibility, this must be a list
#                     dd[obj][par2] = [input_dict[obj][par][i],]
#                 elif isinstance(x, str):
#                     dd[obj][par2] = input_dict[obj][par]
#                 else:
#                     print(x)
#                     raise TypeError('WTF?')
#
#         # Construct objects
#         from pastis.exceptions import EvolTrackError
#         try:
#             objs.append(ob.ObjectBuilder(dd))
#         except EvolTrackError as ex:
#             # Muchos no llegan al final porque los parámetros no
#             # se pueden interpolar
#             print(ex)
#             continue
#
#     #  Construct light curves
#     f = []
#     for obj in objs:
#         # Get period from object list
#         if scenario == 'PLA':
#             P = obj[0].planets[0].orbital_parameters.P
#         elif scenario == 'BEB':
#             P = obj[0].orbital_parameters.P
#
#         # sample according to value of period
#         tess_cadence_min = 2.0
#         n_points = np.int(np.ceil(P * 24 * 60 / tess_cadence_min))
#         tt = np.linspace(0, 1, n_points)
#         from pastis.models import PHOT
#
#         try:
#             lci = PHOT.PASTIS_PHOT(tt, 'TESS',
#                                    True, 0.0, 1.0, 0.0, *obj)
#         except:
#             continue
#
#         f.append([lci, P, n_points])
#
#
#
#
# =============================================================================
