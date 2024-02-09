#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:23:07 2021

Module with functions to randomly draw parameters for given transit scenario
FP or planetary.

@author: rodrigo
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as st

# from pastis import limbdarkening as ld
from pastis.priors import moduleprior as mp
from pastis.MCMC import priors


'''
from . import parameters as pp
from . import constants as cts
'''

import parameters as pp
import constants as cts


class Parameters(object):
    """Global class for paramaters."""

    def __init__(self):
        self.drawn = 0

    def draw(self):
        """Draw all parameters."""
        raise NotImplementedError('draw method must be implemented on a '
                                  'sub-class basis')

    def to_pastis(self, mask=None):
        """
        Prepare dictionary to pass to pastis.

        An array having the same length as the parameters can be drawn
        to keep only certain elements. This is used for transit fitting.
        """
        assert self.draw, "Parameters not drawn. Use .draw method first."

        if mask is None:
            pdict = dict([[par[-1], getattr(self, par[0])] for
                          par in self.parnames.values()])

        else:
            pdict = dict([[par[-1], getattr(self, par[0])[mask]] for
                          par in self.parnames.values()])

        return pdict


class StarParameters(Parameters):
    """General class for abstract stars.

    Only to be used by subclassing.
    """
    def __init__(self):
        self.draw = 0

    def draw_albedo(self):
        """Draw albedo of star."""
        self.albedo = stellar_albedo(len(self))
        return

    def draw_gravitydarkenning(self):
        """Draw gravity darkening coefficient."""
        self.B = np.zeros(len(self))
        return

    def draw_redenning(self):
        """
        Draw value of redenning, E(B-V).

        NOTE: until further notice this just returns 0.
        """
        self.ebmv = np.zeros(len(self))
        return


class TargetStarParameters(StarParameters):
    """class for paeramters of the observed (TIC) target star."""

    def __init__(self, params, pbands=['Johnson-R', ], **kwargs):
        """
        Include parameters from TIC.

        :param array params: shape (nparams+1, size). Array containing the TIC
        parameters of the target stars, ordered column-wise.
        First row contains the effective temperatures.
        Second row holds the metallicity.
        Third row holds the logg / density

        Other parameters are passed to the limb_darkening initialization.
        """
        # TODO: Shall we include distance?
        self.params = params

        # An idea is to include possibility to use density
        # but this is messy with limbdarkening and requires initialisin
        # stellar tracks.
        # self.densflag = params[3]

# =============================================================================
#         # Initialise LDC table (if neeeded)
#         if not hasattr(ld, 'LDCs'):
#             ld.initialize_limbdarkening(pbands, **kwargs)
#             self.inited_pbands = pbands
#         else:
#             self.inited_pbands = list(ld.LDCs[0].keys())
# =============================================================================

        self.parnames = {'Effective Temperature': ['teff', 'K', 'teff'],
                         'Surface gravity': ['logg', 'cgs [log]', 'logg'],
                         'Metallicity': ['feh', '', 'z'],
                         'Distance': ['distance', 'pc', 'dist'],
                         'Albedo': ['albedo', '', 'albedo'],
                         'Redenning': ['ebmv', '', 'ebmv'],
                         'Gravity Darkening': ['B', '', 'B']}

        self.drawn = 0

        return

    @property
    def params(self):
        """Property to return TIC parameters."""
        return dict([['Teff', self.teff], ['logg', self.logg],
                     ['Fe/H', self.feh]])

    @params.setter
    def params(self, params):
        """Property to set TIC parameters."""
        self.teff = params[0]
        self.logg = params[1]
        self.feh = params[2]
        try:
            self.distance = params[3]
        except IndexError:
            self.distance = params[0]*0.0 + 10.0

        self._param_len = params.shape[1]

    def __len__(self):
        """Len method."""
        return self._param_len

    def draw(self):
        """Draw all parameters."""
        self.draw_albedo()
        # self.draw_ldc()
        self.draw_redenning()
        self.draw_gravitydarkenning()

        self.drawn = 1
        return

    def to_pastis(self, *args):
        """Prepare dictionary to pass to PASTIS."""
        # Prepare first set of parameters
        pdict = super().to_pastis(*args)

# =============================================================================
#         # THIS WAS BEFORE, WHEN LDC WERE DRAWN HERE.
#         # But this is useless, as PASTIS will draw them again!
#         # check_pbands and draw_ldc are therefore also commented
#
#         # TODO Fix crash in object builder for LDC dicts.
#         # Make LD coefficients dictionary [not used; for now keep
#         # coefficient from last band iterated]
#         uadict = {}
#         ubdict = {}
#         for band in self.inited_pbands:
#             if hasattr(self, 'LDC_{}'.format(band)):
#                 ldc = getattr(self, 'LDC_{}'.format(band))
#
#                 uadict[band] = ldc[:, 0]
#                 ubdict[band] = ldc[:, 1]
#
#         # pdict.update({'ua': uadict, 'ub': ubdict, 'B': self.feh * 0.0})
#         pdict.update({'ua': ldc[:, 0], 'ub': ldc[:, 1],
#                       'B': self.feh * 0.0})
#
# =============================================================================
# =============================================================================
#         pdict = {'teff': self.teff,
#                  'logg': self.logg,
#                  'z': self.feh,
#                  'albedo': self.albedo,
#                  'ua': uadict,
#                  'ub': ubdict,
#                  # fix beaming effect to zero
#                  'B': self.feh * 0.0
#                  }
# =============================================================================

        return pdict

# =============================================================================
#     def check_pbands(self, pbands):
#         """Check that all elements in a list of photometric bands is initiated."""
#         for p in pbands:
#             if p not in self.inited_pbands:
#                 raise Exception('Sorry; you did not ask for photometric band '
#                                 '{} at instantiation.'.format(p))
#         return
#
# =============================================================================

# =============================================================================
#     def draw_ldc(self, pbands=None):
#         """Draw quadratic limb darkening coefficients for target stars."""
#         if pbands is not None:
#             # Check pbands
#             self.check_pbands(pbands)
#         else:
#             pbands = self.inited_pbands
#
#         for band in pbands:
#             ldc_p = [ld.get_LD(teff, logg, feh, band) for
#                      teff, logg, feh in zip(self.teff, self.logg, self.feh)]
#             setattr(self, 'LDC_{}'.format(band), np.array(ldc_p))
#
#         return
# =============================================================================


class PlanetParameters(Parameters):
    """class of realistic parameters for the planet scenario."""

    def __init__(self, 
                 minradius=None,
                 max_period=pp.MAX_PERIOD, 
                 method='hsu',
                 table_path=os.path.join(pp.TABLE_DIR, 'Hsu',
                                               'table2.dat'),
                 rates_column=4, 
                 interbindist='flat', 
                 **kwargs):
        """
        If method is 'hsu', prepare rates from Hsu+2019 table 2.

        This table contains the occurrence rates measured in two different
        ways (cols. 4 and 6) for each bin of period (cols. 0 and 1) and
        planet size (cols. 2 and 3)

        :param float or None minradius: minimum radius to consider. If None
        no limit is included.

        :params str method: whether to use "Hsu table" ('hsu') or uniform distributions ('uniform').
        
        :param int rates_column: chooses which rate to use. Options 4 or 6

        :param str interbindist: defines method to sample within bin. Options
        are "flat" or "logflat"

        The remaning parameters are passed to the draw functions.
        """
        validmethod = ['hsu', 'uniform']
        
        assert method in validmethod, \
            ("method must be \'{}\'".format('\' or \''.join(validmethod)))
        self.method = method
        
        validdist = ['flat', 'logflat']

        assert interbindist in validdist, \
            ("interbindist must be \'{}\'".format('\' or \''.join(validdist)))
        self.sampledist = interbindist

        dd = pd.read_csv(table_path, delim_whitespace=True, header=None,
                         index_col=None)

        self.rates_column = rates_column

        # Remove upper limits (always use column 4, because 9 may be NaN)
        self.occ_rate_table = dd.loc[dd.loc[:, 4] != "<"]

        # Get occurrence rate in numpy format to use as weights
        self.rates_orig = self.occ_rate_table.loc[:, self.rates_column].\
            to_numpy().astype('float')

        if minradius is not None:
            min_rad_cond = self.occ_rate_table.loc[:, 2] >= minradius
        else:
            min_rad_cond = np.full_like(self.occ_rate_table.loc[:, 2], True)

        if max_period is not None:
            max_per_cond = self.occ_rate_table.loc[:, 1] <= max_period
        else:
            max_per_cond = np.full_like(self.occ_rate_table.loc[:, 1], True)

        # Filter rates
        # TODO allow filtering inside bin; up to now, only full bins are kept
        self.rates = np.where(min_rad_cond * max_per_cond, self.rates_orig, 0)

        # Compute weights
        self.w = self.rates/self.rates.sum()

        # Parameter names
        # Each entry contains name of attribute, units
        # and name of parameter in pastis

        self.parnames = {'orb_period': ['period', 'days', 'P'],
                         'orb_ecc': ['ecc', '', 'ecc'],
                         'orb_omega': ['omega_rad', 'rad', 'omega'],
                         'orb_incl': ['incl_rad', 'rad', 'incl'],
                         'orb_phtr': ['ph_tr', '', 'T0'],
                         'pla_radius_rjup': ['radius_rjup', 'jupiter', 'Rp'],
                         'pla_mass_mjup': ['mass_mjup', 'jupiter', 'Mp'],
                         'pla_albedo': ['albedo', 'albedo']}

        for val in self.parnames.values():
            setattr(self, val[0], None)
        self.drawn = 0
        return

    def draw(self, size=1, **kwargs):
        """Draw all parameters at once (convenience function)."""
        self.draw_orbit(size, **kwargs)
        self.draw_period_radius(size, **kwargs)
        self.draw_mass(size)
        self.draw_albedo(size)

        self.drawn = 1
        return

    def to_pastis(self, *args):
        """
        Prepare dictionary to pass to PASTIS.

        A number of unit transformations are needed to achieve this.
        """
        pdict = super().to_pastis(*args)

        # pass omega and incl to degrees, as required in pastis
        for angle in ['omega', 'incl']:
            pdict[angle] *= 180/np.pi

        return pdict

    def draw_period_radius(self, size=1, **kwargs):
        """Draw size parameters following the planet occurrence rates."""
        
        if self.method == 'hsu':
            
            # Randomly draw a line with weights self.w
            i = np.random.choice(len(self.w), size=size, p=self.w)

            A = self.occ_rate_table.iloc[i, [0, 1, 2, 3]].to_numpy()

            deltap = A[:, 1] - A[:, 0]
            deltar = A[:, 3] - A[:, 2]
            
            pmin = A[:, 0]
            rmin = A[:, 2]

        elif self.method == 'uniform':
            
            # Get all columns not filtered in __init__
            A = self.occ_rate_table.loc[self.w > 0, [0, 1, 2, 3]].to_numpy()
                        
            # Rango en períodos y radios
            deltap = A[:, 1].max() - A[:, 0].min()
            deltar = A[:, 3].max() - A[:, 2].min()
            
            pmin = A[:, 0].min()
            rmin = A[:, 2].min()      
            
        # Sample randomly within bin
        u = np.random.rand(size, 2)

        if self.sampledist == 'flat':
            p = u[:, 0] * deltap + pmin
            r = u[:, 1] * deltar + rmin

        elif self.sampledist == 'logflat':
            # TDOO draw log-flat
            pass

        self.period = p
        self.radius_rearth = r
        self.radius_rjup = r * cts.Rearth / cts.Rjup

        return

    def draw_mass(self, size=1):
        """Draw planet mass."""
        try:
            r = self.radius_rearth
        except AttributeError:
            raise AttributeError('Planet radius not defined; '
                                 'run draw_period_radius first')

        # TODO realistic mass-radius relation
        self.mass_mearth = r*0.0 + 1.0
        self.mass_mjup = self.mass_mearth * cts.GMearth / cts.GMjup

        return

    def draw_albedo(self, size=1):
        """Draw planet albedo."""
        self.albedo = np.random.rand(size)
        return

    def draw_orbit(self, size=1, **kwargs):
        """Draw orbital parameters, except period."""

        thetamin_deg = kwargs.pop('thetamin_deg', pp.THETAMIN_DEG)
        eccentric = kwargs.pop('eccentric', True)

        # Random inclination between thetamin and 90.0 deg
        k = np.cos(thetamin_deg * np.pi/180.0)
        self.incl_rad = np.arccos(k * (1 - np.random.rand(size)))

        # transit phase
        self.ph_tr = np.random.rand(size)

        # Eccentricity
        if eccentric:
            ecc_prior = priors.TruncatedUNormalPrior(0., 0.3, 0., 1.)
            self.ecc = ecc_prior.rvs(size)
            # self.ecc = np.abs(np.random.randn(size) * 0.3)
            self.omega_rad = np.random.rand(size) * 2 * np.pi

        else:
            self.ecc = np.array([0]*size)
            self.omega_rad = np.array([0]*size)

        # For future convenience, define omega in deg.
        self.omega_deg = self.omega_rad * 180.0 / np.pi

        return


class BlendedStarParameters(StarParameters):
    """General class for stars blended by a Target star."""

    def __init__(self, foreground, minmass=0.05, *args):
        """
        Instatiate class.

        :param TargetStar foreground: star in the foreground
        :param float maxdist: maximum distance [in pc] at which a background
        star is allowed to be located.
        :param float minmass: minimum mass allowed for background stars.

        Parameters could include galactic direction, redenning, etc.
        """
        self.foreground = foreground
        self.minmass = minmass

        self.parnames = {'Mass': ['mass', 'Solar mass', 'minit'],
                         'Age': ['logage', 'Gyr [log]', 'logage'],
                         'Metallicity': ['feh', '', 'z'],
                         'Albedo': ['albedo', '', 'albedo'],
                         'Redenning': ['ebmv', '', 'ebmv']
                         }
        self.drawn = 0

        return

    def __len__(self):
        """Len method."""
        return len(self.foreground)

    def draw(self):
        """Draw all parameters. Convenience function."""
        for d in ['mass', 'logage', 'feh', 'distance', 'albedo', 'redenning']:
            to_run = getattr(self, 'draw_{}'.format(d))
            # Run draw
            to_run()

        self.drawn = 1

        return

    def draw_mass(self):
        """
        Draw mass of background star.

        Use IMF from Robin+2003
        """
        # Parameters from Robin+2003
        alpha = 1.6
        beta = 3.0
        m0 = 1.0

        amax = m0**(1 - alpha) / (1 - alpha)
        amin = self.minmass**(1 - alpha) / (1 - alpha)

        # bmax = self.maxmass**(1 - beta) / (1 - beta)
        bmin = m0**(1 - beta) / (1 - beta)

        # Integral over whole range
        # k = amax - amin + bmax - bmin
        k = amax - amin - bmin

        # Limit quantiles
        q0 = (amax - amin)/k

        # Random quantile draws
        q = np.random.rand(len(self))

        # Inverse CDF in the two regimes
        xalpha = ((q*k + amin) * (1 - alpha)) ** (1 / (1 - alpha))
        xbeta = ((q*k - (amax - amin) + bmin) * (1 - beta)) ** (1 / (1 - beta))

        # Assign depending on value of q
        self.mass = np.where(q < q0, xalpha, xbeta)

        return

    def draw_logage(self):
        """Draw logarithm (base 10) of age in Gyr."""
        self.logage = np.random.rand(len(self))*4 + 6
        return

    def draw_feh(self):
        """Draw Fe/H metallicity."""
        self.feh = np.random.rand(len(self)) * 3 - 2.5
        return

    def draw_distance(self):
        """Draw distance of blended star (with respecto to target star)."""
        raise NotImplementedError('draw_distance method must be implented '
                                  'by subclass')


# TODO Clean up; most methods are identically to BlendedStarParameters
# Leave draw_distance
class BackgroundStarParameters(BlendedStarParameters):
    """Class with realistic parameters for background stars."""

    def __init__(self, foreground, maxdist=1000, minmass=0.05, *args):
        """
        Instatiate class.

        :param TargetStar foreground: star in the foreground
        :param float maxdist: maximum distance [in pc] at which a background
        star is allowed to be located.
        :param float minmass: minimum mass allowed for background stars.

        Parameters could include galactic direction, redenning, etc.
        """

        super().__init__(foreground, minmass=minmass)

        # Add information related to distance behind target star
        self.maxdist = maxdist
        self.parnames.update({'Distance': ['distance', 'pc', 'dist'],})

        return

    def draw(self):
        """Draw all parameters. Convenience function."""
        for d in ['mass', 'logage', 'feh', 'distance', 'albedo', 'redenning']:
            to_run = getattr(self, 'draw_{}'.format(d))
            # Run draw
            to_run()

        self.drawn = 1

        return

    def draw_mass(self):
        """
        Draw mass of background star.

        Use IMF from Robin+2003
        """
        # Parameters from Robin+2003
        alpha = 1.6
        beta = 3.0
        m0 = 1.0

        amax = m0**(1 - alpha) / (1 - alpha)
        amin = self.minmass**(1 - alpha) / (1 - alpha)

        # bmax = self.maxmass**(1 - beta) / (1 - beta)
        bmin = m0**(1 - beta) / (1 - beta)

        # Integral over whole range
        # k = amax - amin + bmax - bmin
        k = amax - amin - bmin

        # Limit quantiles
        q0 = (amax - amin)/k

        # Random quantile draws
        q = np.random.rand(len(self))

        # Inverse CDF in the two regimes
        xalpha = ((q*k + amin) * (1 - alpha)) ** (1 / (1 - alpha))
        xbeta = ((q*k - (amax - amin) + bmin) * (1 - beta)) ** (1 / (1 - beta))

        # Assign depending on value of q
        self.mass = np.where(q < q0, xalpha, xbeta)

        return

    def draw_logage(self):
        """Draw logarithm (base 10) of age in Gyr."""
        self.logage = np.random.rand(len(self))*4 + 6
        return

    def draw_feh(self, size=1):
        """Draw Fe/H metallicity."""
        self.feh = np.random.rand(len(self)) * 3 - 2.5
        return

    def draw_distance(self, size=1):
        """Draw distance of background star (from target star)."""
        # TODO consider foreground stars.
        self.distance = np.random.rand(len(self))**(1./3.) * self.maxdist
        # Add distance to foreground star
        self.distance += self.foreground.distance
        return


class PrimaryBkgParameters(BackgroundStarParameters):
    """Class for parameters of primary background star."""

    def __init__(self, foreground, maxdist=1000, minmass=0.05):

        super().__init__(foreground, maxist=maxdist,
                         minmass=minmass)
        return


class SecondaryBkgParameters(BlendedStarParameters):
    """Class for parameters of secondary background star."""

    def __init__(self, primarystar):

        assert primarystar.drawn, "Undrawn primary star parameters. Abort."

        self.primary = primarystar

        super().__init__(primarystar, minmass=primarystar.minmass)

        return

    def draw_q(self):
        """
        Draw mass ratio.

        Use statistics by Raghavan et al. (2010; their fig. 16, left panel).

        See moduleprior.q_def for more details.
        """
        self.q = binary_mass_ratio(len(self))
        return

    def draw_mass(self):
        """Draw mass based on mass ratio and primary mass."""
        # Draw q
        if not hasattr(self, 'q'):
            self.draw_q()

        self.mass = self.primary.mass * self.q
        return

    def draw(self):
        """Draw all parameters. Convenience function."""
        for d in ['mass', 'albedo', 'redenning']:
            to_run = getattr(self, 'draw_{}'.format(d))
            # Run draw
            to_run()

        # Copy specific attributes from primary star
        # This is redundant as PASTIS takes care of this anyway
        for inherited_par in ['logage', 'feh']:
            setattr(self, inherited_par, getattr(self.primary, inherited_par))

        # Distance from primary
        self.distance = self.primary.distance * 0.0

        self.drawn = 1

        return


class SecondaryStarParameters(StarParameters):
    """Class for parameters of secondary star bound to target."""

    def __init__(self, primarystar):

        assert primarystar.drawn, "Undrawn primary star parameters. Abort."

        self.primary = primarystar

        self.parnames = {'q': ['q', '', 'q'],
                         'Albedo': ['albedo', '', 'albedo2'],
                         }
        self.drawn = 0

        return

    def __len__(self):
        """Len method."""
        return len(self.primary)

    def draw_q(self):
        """
        Draw mass ratio.

        Use statistics by Raghavan et al. (2010; their fig. 16, left panel).

        See moduleprior.q_def for more details.
        """
        self.q =binary_mass_ratio(len(self))
        return


    def draw(self):
        """Draw all parameters. Convenience function."""
        for d in ['q', 'albedo']:
            to_run = getattr(self, 'draw_{}'.format(d))
            # Run draw
            to_run()

        # Copy specific attributes from primary star
        # This is redundant as PASTIS takes care of this anyway
        """
        for inherited_par in ['feh', 'distance']:
            setattr(self, inherited_par, getattr(self.primary, inherited_par))
        """
        self.drawn = 1

        return


class BoundPrimaryParameters(StarParameters):
    """Class with realistic parameters for background stars."""

    def __init__(self, foreground, minmass=0.05, *args):
        """
        Instatiate class.

        :param TargetStar foreground: primary star in hierarchichal triple
        :param float minmass: minimum mass allowed for binary components.

        Parameters could include galactic direction, redenning, etc.
        """
        self.foreground = foreground
        self.minmass = minmass

        self.parnames = {'Mass': ['mass', 'Solar mass', 'minit'],
                         'Age': ['logage', 'Gyr [log]', 'logage'],
                         'Metallicity': ['feh', '', 'z'],
                         'Distance': ['distance', 'pc', 'dist'],
                         'Albedo': ['albedo', '', 'albedo'],
                         'Redenning': ['ebmv', '', 'ebmv']
                         }
        self.drawn = 0

        return

    def __len__(self):
        """Len method."""
        return len(self.foreground)

    def draw(self):
        """Draw all parameters. Convenience function."""
        for d in ['mass', 'feh', 'logage', 'distance', 'albedo', 'redenning']:
            to_run = getattr(self, 'draw_{}'.format(d))
            # Run draw
            to_run()

        # Distance from foreground
        # TODO check DISTANCES! Are they relative to target or not!?
        # self.distance = self.foreground.distance * 0.0

        self.drawn = 1

        return

    def draw_mass(self):
        """
        Draw mass of background star.

        Use IMF from Robin+2003
        """
        #TODO change to something more clever (tokovinin or Raghavan)

        # Parameters from Robin+2003
        alpha = 1.6
        beta = 3.0
        m0 = 1.0

        amax = m0**(1 - alpha) / (1 - alpha)
        amin = self.minmass**(1 - alpha) / (1 - alpha)

        # bmax = self.maxmass**(1 - beta) / (1 - beta)
        bmin = m0**(1 - beta) / (1 - beta)

        # Integral over whole range
        # k = amax - amin + bmax - bmin
        k = amax - amin - bmin

        # Limit quantiles
        q0 = (amax - amin)/k

        # Random quantile draws
        q = np.random.rand(len(self))

        # Inverse CDF in the two regimes
        xalpha = ((q*k + amin) * (1 - alpha)) ** (1 / (1 - alpha))
        xbeta = ((q*k - (amax - amin) + bmin) * (1 - beta)) ** (1 / (1 - beta))

        # Assign depending on value of q
        self.mass = np.where(q < q0, xalpha, xbeta)

        return

    def draw_feh(self):
        """Use Fe/H from target star."""
        self.feh = self.foreground.feh
        return

    def draw_logage(self):
        """Use logage from target star."""
        try:
            self.logage = self.foreground.logage
        except AttributeError:
            # It does not really matter what age we assign at this point
            # PASTIS will correct this later
            self.logage = np.array([4,]*len(self))
        return

    def draw_distance(self):
        """Use distance from target star."""
        self.distance = self.foreground.distance
        return


class OrbitParameters(Parameters):
    """Class of orbital parameters (except periods)."""

    def __init__(self, orbittype='planet', **kwargs):

        max_period = kwargs.pop('max_period', pp.MAX_PERIOD)

        valid_types = ['planet', 'binary', 'triple']

        assert orbittype in valid_types, \
            ("orbittype must be \'{}\'".format('\' or \''.join(valid_types)))

        self.type = orbittype
        self.max_period = max_period

        self.parnames = {'orb_ecc': ['ecc', '', 'ecc'],
                         'orb_omega': ['omega_rad', 'rad', 'omega'],
                         'orb_incl': ['incl_rad', 'rad', 'incl'],
                         'orb_phtr': ['ph_tr', '', 'T0'],
                         'orb_period': ['period', 'P'],
                         }
        return

    def draw(self, size, **kwargs):
        """
        Draw all parameters.

        There is a small difference if it is a binary or a planet
        orbit.
        """

        thetamin = kwargs.pop('thetamin_deg', pp.THETAMIN_DEG)

        if self.type == 'planet':
            #TODO this is not working!
            self.draw_orbit(size, thetamin_deg=thetamin)

        elif self.type == 'binary':

            if self.max_period is None:
                self.log_period = np.random.randn(size) * 2.28 + 5.03
            else:
                a_ = -np.inf
                b_ = (np.log(self.max_period) - 5.03)/2.28
                self.log_period = st.truncnorm.rvs(a=a_, b=b_,
                                                   loc=5.03, scale=2.28, size=size)
            self.period = np.exp(self.log_period)

            self.draw_angles_phase(size, thetamin_deg=thetamin)

        elif self.type == 'triple':
            # Very long period, does not really affect as long
            # as much longer than shorter period in triple
            self.period = np.full(size, 10000.0)
            self.log_period = np.log(self.period)

            self.draw_angles_phase(size, thetamin_deg=0.0)

        else:
            raise ValueError('Unknown object type')
        return

    def to_pastis(self, *args):
        """
        Specific version of to_pastis for orbits.

        Pastis requires angles in degrees, so transformation is
        necessary.
        """
        pdict = super().to_pastis(*args)

        # pass omega and incl to degrees, as required in pastis
        for angle in ['omega', 'incl']:
            pdict[angle] *= 180/np.pi

        return pdict

    def draw_angles_phase(self, size=1, thetamin_deg=60.0, eccentric=True):
        """Draw orbital angles and phase."""
        # Random inclination between thetamin and 90.0 deg
        k = np.cos(thetamin_deg * np.pi/180.0)
        self.incl_rad = np.arccos(k * (1 - np.random.rand(size)))

        # transit phase
        self.ph_tr = np.random.rand(size)

        # Eccentricity
        if eccentric:
            ecc_prior = priors.TruncatedUNormalPrior(0., 0.3, 0., 1.)
            self.ecc = ecc_prior.rvs(size)
            self.omega_rad = np.random.rand(size) * 2 * np.pi

        else:
            self.ecc = np.array([0]*size)
            self.omega_rad = np.array([0]*size)

        # For future convenience, define omega in deg.
        self.omega_deg = self.omega_rad * 180.0 / np.pi

        return


def stellar_albedo(size):
    """Random sample of stellar albedos."""
    #TODO include reference for albedo values
    return np.random.rand(size) * 0.4 + 0.6


def binary_mass_ratio(size):
    """Draw mass ratio q."""
    # TODO: use joint (q, primary mass) distribution instead of marginal

    q = np.empty(size)
    # TODO: awful; try to vectorize function in moduleprior
    for i in range(len(q)):
        q[i] = mp.q_def()
        
    return q