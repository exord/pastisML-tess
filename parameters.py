"""
Define parameters for simulations.

Created on Fri May  7 17:34:03 2021

@author: rodrigo
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
SCENARIO = 'PLA'
NSIMU_PER_TIC_STAR = 1

# Directory with tables
TABLE_DIR = 'tables/'

# Limits on radius
MIN_PLA_RADIUS = 1.0

# Limits on diluted planets
MIN_DILUTED_PLANET_RADIUS = 8.0

# Maximum magnitude difference between target and blended system
MAX_MAG_DIFF = 8.0

# Minimum depth for transit / eclipse in parts-per-millon
MIN_DEPTH = 300

# Maximum period
MAX_PERIOD = 16

# Lower bound to inclination angle
THETAMIN_DEG = 85.0

# Maximum distance for blended stars in pc
MAX_DIST = 200.0