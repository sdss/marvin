#!/usr/bin/env python
# encoding: utf-8
#
# MPL5.py
#
# Created by José Sánchez-Gallego on 8 Aug 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from astropy import units as u

from .base import Bintype, Template, DAPDataModel, Property, MultiChannelProperty, spaxel
from .MPL4 import MPL4_emline_channels


GAU_MILESHC = Template('GAU-MILESHC')

ALL = Bintype('ALL')
NRE = Bintype('NRE')
SPX = Bintype('SPX', binned=False)
VOR10 = Bintype('VOR10')


MPL5_extra_channels = ['oii_3727', 'oii_3729', 'heps_3971', 'hdel_4102', 'hgam_4341', 'heii_4687',
                       'hei_5877', 'siii_8831', 'siii_9071', 'siii_9533']

MPL5_maps = [
    MultiChannelProperty('spx_skycoo', ivar=False, mask=False, channels=['on_sky_x', 'on_sky_y'],
                         units=u.arcsec,
                         description='Offsets of each spaxel from the galaxy center.'),
    MultiChannelProperty('spx_ellcoo', ivar=False, mask=False,
                         channels=['elliptical_radius', 'elliptical_azimuth'],
                         units=[u.arcsec, u.deg],
                         description='Elliptical polar coordinates of each spaxel from '
                                     'the galaxy center.'),
    Property('spx_mflux', ivar=True, mask=False,
             unit=u.erg / u.s / (u.cm ** 2) / spaxel, scale=1e-17,
             description='Mean flux in r-band (5600.1-6750.0 ang).'),
    Property('spx_snr', ivar=False, mask=False,
             description='r-band signal-to-noise ratio per pixel.'),
    Property('binid', ivar=False, mask=False, unit=None,
             description='Numerical ID for spatial bins.'),
    MultiChannelProperty('bin_lwskycoo', ivar=False, mask=False,
                         channels=['lum_weighted_on_sky_x', 'lum_weighted_on_sky_y'],
                         units=u.arcsec,
                         description='Light-weighted offset of each bin from the galaxy center.'),
    MultiChannelProperty('bin_lwellcoo', ivar=False, mask=False,
                         channels=['lum_weighted_elliptical_radius',
                                   'lum_weighted_elliptical_azimuth'],
                         units=[u.arcsec, u.deg],
                         description='Light-weighted elliptical polar coordinates of each bin '
                                     'from the galaxy center.'),
    Property('bin_area', ivar=False, mask=False,
             unit=u.arcsec ** 2,
             description='Area of each bin.'),
    Property('bin_farea', ivar=False, mask=False,
             unit=None,
             description='Fractional area that the bin covers for the expected bin '
                         'shape (only relevant for radial binning).'),
    Property('bin_mflux', ivar=True, mask=True,
             unit=u.erg / u.s / (u.cm ** 2) / spaxel, scale=1e-17,
             description='Mean flux in the r-band for the binned spectra.'),
    Property('bin_snr', ivar=False, mask=False,
             unit=None,
             description='r-band signal-to-noise ratio per pixel in the binned spectra.'),
    Property('stellar_vel', ivar=True, mask=True,
             unit=u.km / u.s,
             description='Stellar velocity relative to NSA redshift.'),
    Property('stellar_sigma', ivar=True, mask=True,
             unit=u.km / u.s,
             description='Stellar velocity dispersion (must be corrected using '
                         'STELLAR_SIGMACORR)'),
    Property('stellar_sigmacorr', ivar=False, mask=False,
             unit=u.km / u.s,
             description='Quadrature correction for STELLAR_SIGMA to obtain the '
                         'astrophysical velocity dispersion.)'),
    MultiChannelProperty('stellar_cont_fresid', ivar=False, mask=False,
                         channels=['68th_percentile', '99th_percentile'],
                         units=None,
                         description='68%% and 99%% growth of the fractional residuals between '
                                     'the model and data'),
    Property('stellar_cont_rchi2', ivar=False, mask=False,
             unit=None,
             description='Reduced chi-square of the stellar continuum fit.'),
    MultiChannelProperty('emline_sflux', ivar=True, mask=True,
                         channels=MPL4_emline_channels + MPL5_extra_channels,
                         units=u.erg / u.s / (u.cm ** 2) / spaxel, scales=1e-17,
                         description='Non-parametric summed flux for emission lines.'),
    MultiChannelProperty('emline_sew', ivar=True, mask=True,
                         channels=MPL4_emline_channels + MPL5_extra_channels,
                         units=u.Angstrom,
                         description='Emission line non-parametric equivalent '
                                     'widths measurements.'),
    MultiChannelProperty('emline_gflux', ivar=True, mask=True,
                         channels=MPL4_emline_channels + MPL5_extra_channels,
                         units=u.erg / u.s / (u.cm ** 2) / spaxel, scales=1e-17,
                         description='Gaussian profile integrated flux for emission lines.'),
    MultiChannelProperty('emline_gvel', ivar=True, mask=True,
                         channels=MPL4_emline_channels + MPL5_extra_channels,
                         units=u.km / u.s,
                         description='Gaussian profile velocity for emission lines.'),
    MultiChannelProperty('emline_gsigma', ivar=True, mask=True,
                         channels=MPL4_emline_channels + MPL5_extra_channels,
                         units=u.km / u.s,
                         description='Gaussian profile velocity dispersion for emission lines; '
                                     'must be corrected using EMLINE_INSTSIGMA'),
    MultiChannelProperty('emline_instsigma', ivar=False, mask=False,
                         channels=MPL4_emline_channels + MPL5_extra_channels,
                         units=u.km / u.s,
                         description='Instrumental dispersion at the fitted line center.'),
    MultiChannelProperty('specindex', ivar=True, mask=True,
                         channels=['d4000', 'dn4000'],
                         units=None,
                         description='Measurements of spectral indices.'),
    MultiChannelProperty('specindex_corr', ivar=False, mask=False,
                         channels=['d4000', 'dn4000'],
                         units=None,
                         description='Velocity dispersion corrections for the '
                                     'spectral index measurements '
                                     '(can be ignored for D4000, Dn4000).')
]


MPL5 = DAPDataModel('2.0.2', aliases=['MPL-5'], bintypes=[ALL, NRE, VOR10, SPX],
                    templates=[GAU_MILESHC], properties=MPL5_maps,
                    default_bintype='SPX', default_template='GAU-MILESHC')
