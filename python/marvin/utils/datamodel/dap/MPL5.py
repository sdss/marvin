#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-08-08
# @Filename: MPL5.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-30 11:45:02


from __future__ import absolute_import, division, print_function

from astropy import units as u

from marvin.utils.datamodel.maskbit import get_maskbits

from .base import (Bintype, Channel, DAPDataModel, Model,
                   MultiChannelProperty, Property, Template, spaxel)
from .MPL4 import MPL4_emline_channels


GAU_MILESHC = Template('GAU-MILESHC',
                       description='Set of stellar templates derived from the MILES library.')

ALL = Bintype('ALL', description='Sum of all spectra in each datacube.')
NRE = Bintype('NRE', description='Two radial bins, binning all spectra from 0-1 '
                                 'and 1-2 (elliptical Petrosian) effective radii.')
SPX = Bintype('SPX', binned=False, description='Unbinned spaxels.')
VOR10 = Bintype('VOR10', description='Spectra binned to S/N~10 using the Voronoi '
                                     'binning algorithm (Cappellari & Copin 2003).')


binid_property = Property('binid', ivar=False, mask=False,
                          formats={'string': 'Bin ID'},
                          description='ID number for the bin for which the pixel value was '
                                      'calculated; bins are sorted by S/N.')


last_idx = len(MPL4_emline_channels) - 1
MPL5_emline_channels = MPL4_emline_channels + [
    Channel('oii_3727', formats={'string': 'OII 3727',
                                 'latex': r'$\forb{O\,II}\;\lambda 3727$'}, idx=last_idx + 1),
    Channel('oii_3729', formats={'string': 'OII 3729',
                                 'latex': r'$\forb{O\,II}\;\lambda 3729$'}, idx=last_idx + 2),
    Channel('heps_3971', formats={'string': 'H-epsilon 3971',
                                  'latex': r'H$\epsilon\;\lambda 3971$'}, idx=last_idx + 3),
    Channel('hdel_4102', formats={'string': 'H-delta 4102',
                                  'latex': r'H$\delta\;\lambda 4102$'}, idx=last_idx + 4),
    Channel('hgam_4341', formats={'string': 'H-gamma 4341',
                                  'latex': r'H$\gamma\;\lambda 4341$'}, idx=last_idx + 5),
    Channel('heii_4687', formats={'string': 'HeII 4681',
                                  'latex': r'He\,II$\;\lambda 4687$'}, idx=last_idx + 6),
    Channel('hei_5877', formats={'string': 'HeI 5877',
                                 'latex': r'He\,I$\;\lambda 5877$'}, idx=last_idx + 7),
    Channel('siii_8831', formats={'string': 'SIII 8831',
                                  'latex': r'$\forb{S\,III}\;\lambda 8831$'}, idx=last_idx + 8),
    Channel('siii_9071', formats={'string': 'SIII 9071',
                                  'latex': r'$\forb{S\,III}\;\lambda 9071$'}, idx=last_idx + 9),
    Channel('siii_9533', formats={'string': 'SIII 9533',
                                  'latex': r'$\forb{S\,III}\;\lambda 9533$'}, idx=last_idx + 10)
]

MPL5_specindex_channels = [
    Channel('d4000', formats={'string': 'D4000'}, unit=u.dimensionless_unscaled, idx=0),
    Channel('dn4000', formats={'string': 'Dn4000'}, unit=u.dimensionless_unscaled, idx=1)
]


MPL5_maps = [
    MultiChannelProperty('spx_skycoo', ivar=False, mask=False,
                         channels=[Channel('on_sky_x', formats={'string': 'On-sky X'}, idx=0),
                                   Channel('on_sky_y', formats={'string': 'On-sky Y'}, idx=1)],
                         unit=u.arcsec,
                         formats={'string': 'Sky coordinates'},
                         description='Offsets of each spaxel from the galaxy center.'),
    MultiChannelProperty('spx_ellcoo', ivar=False, mask=False,
                         channels=[Channel('elliptical_radius',
                                           formats={'string': 'Elliptical radius'},
                                           idx=0, unit=u.arcsec),
                                   Channel('elliptical_azimuth',
                                           formats={'string': 'Elliptical azimuth'},
                                           idx=1, unit=u.deg)],
                         formats={'string': 'Elliptical coordinates'},
                         description='Elliptical polar coordinates of each spaxel from '
                                     'the galaxy center.'),
    Property('spx_mflux', ivar=True, mask=False,
             unit=u.erg / u.s / (u.cm ** 2) / spaxel, scale=1e-17,
             formats={'string': 'r-band mean flux'},
             description='Mean flux in r-band (5600.1-6750.0 ang).'),
    Property('spx_snr', ivar=False, mask=False,
             formats={'string': 'r-band SNR'},
             description='r-band signal-to-noise ratio per pixel.'),
    binid_property,
    MultiChannelProperty('bin_lwskycoo', ivar=False, mask=False,
                         channels=[Channel('lum_weighted_on_sky_x',
                                           formats={'string': 'Light-weighted offset X'},
                                           idx=0, unit=u.arcsec),
                                   Channel('lum_weighted_on_sky_y',
                                           formats={'string': 'Light-weighted offset Y'},
                                           idx=1, unit=u.arcsec)],
                         description='Light-weighted offset of each bin from the galaxy center.'),
    MultiChannelProperty('bin_lwellcoo', ivar=False, mask=False,
                         channels=[Channel('lum_weighted_elliptical_radius',
                                           formats={'string': 'Light-weighted radial offset'},
                                           idx=0, unit=u.arcsec),
                                   Channel('lum_weighted_elliptical_azimuth',
                                           formats={'string': 'Light-weighted azimuthal offset'},
                                           idx=1, unit=u.deg)],
                         description='Light-weighted elliptical polar coordinates of each bin '
                                     'from the galaxy center.'),
    Property('bin_area', ivar=False, mask=False,
             unit=u.arcsec ** 2,
             formats={'string': 'Bin area'},
             description='Area of each bin.'),
    Property('bin_farea', ivar=False, mask=False,
             formats={'string': 'Bin fractional area'},
             description='Fractional area that the bin covers for the expected bin '
                         'shape (only relevant for radial binning).'),
    Property('bin_mflux', ivar=True, mask=True,
             unit=u.erg / u.s / (u.cm ** 2) / spaxel, scale=1e-17,
             formats={'string': 'r-band binned spectra mean flux'},
             description='Mean flux in the r-band for the binned spectra.'),
    Property('bin_snr', ivar=False, mask=False,
             formats={'string': 'Bin SNR'},
             description='r-band signal-to-noise ratio per pixel in the binned spectra.'),
    Property('stellar_vel', ivar=True, mask=True,
             unit=u.km / u.s,
             formats={'string': 'Stellar velocity'},
             description='Stellar velocity relative to NSA redshift.'),
    Property('stellar_sigma', ivar=True, mask=True,
             unit=u.km / u.s,
             formats={'string': 'Stellar velocity dispersion', 'latex': r'Stellar $\sigma$'},
             description='Stellar velocity dispersion (must be corrected using '
                         'STELLAR_SIGMACORR)'),
    Property('stellar_sigmacorr', ivar=False, mask=False,
             unit=u.km / u.s,
             formats={'string': 'Stellar sigma correction',
                      'latex': r'Stellar $\sigma$ correction'},
             description='Quadrature correction for STELLAR_SIGMA to obtain the '
                         'astrophysical velocity dispersion.)'),
    MultiChannelProperty('stellar_cont_fresid', ivar=False, mask=False,
                         channels=[Channel('68th_percentile',
                                           formats={'string': '68th percentile',
                                                    'latex': r'68^{th} percentile'}, idx=0),
                                   Channel('99th_percentile',
                                           formats={'string': '99th percentile',
                                                    'latex': r'99^{th} percentile'}, idx=1)],
                         formats={'string': 'Fractional residual growth'},
                         description='68%% and 99%% growth of the fractional residuals between '
                                     'the model and data.'),
    Property('stellar_cont_rchi2', ivar=False, mask=False,
             formats={'string': 'Stellar continuum reduced chi-square',
                      'latex': r'Stellar\ continuum\ reduced\ \chi^2'},
             description='Reduced chi-square of the stellar continuum fit.'),
    MultiChannelProperty('emline_sflux', ivar=True, mask=True,
                         channels=MPL5_emline_channels,
                         formats={'string': 'Emission line summed flux'},
                         unit=u.erg / u.s / (u.cm ** 2) / spaxel, scale=1e-17,
                         description='Non-parametric summed flux for emission lines.'),
    MultiChannelProperty('emline_sew', ivar=True, mask=True,
                         channels=MPL5_emline_channels,
                         formats={'string': 'Emission line EW'},
                         unit=u.Angstrom,
                         description='Emission line non-parametric equivalent '
                                     'widths measurements.'),
    MultiChannelProperty('emline_gflux', ivar=True, mask=True,
                         channels=MPL5_emline_channels,
                         formats={'string': 'Emission line Gaussian flux'},
                         unit=u.erg / u.s / (u.cm ** 2) / spaxel, scale=1e-17,
                         description='Gaussian profile integrated flux for emission lines.'),
    MultiChannelProperty('emline_gvel', ivar=True, mask=True,
                         channels=MPL5_emline_channels,
                         formats={'string': 'Emission line Gaussian velocity'},
                         unit=u.km / u.s,
                         description='Gaussian profile velocity for emission lines.'),
    MultiChannelProperty('emline_gsigma', ivar=True, mask=True,
                         channels=MPL5_emline_channels,
                         formats={'string': 'Emission line Gaussian sigma',
                                  'latex': r'Emission line Gaussian $\sigma$'},
                         unit=u.km / u.s,
                         description='Gaussian profile velocity dispersion for emission lines; '
                                     'must be corrected using EMLINE_INSTSIGMA.'),
    MultiChannelProperty('emline_instsigma', ivar=False, mask=False,
                         channels=MPL5_emline_channels,
                         formats={'string': 'Emission line instrumental sigma',
                                  'latex': r'Emission line instrumental $\sigma$'},
                         unit=u.km / u.s,
                         description='Instrumental dispersion at the fitted line center.'),
    MultiChannelProperty('specindex', ivar=True, mask=True,
                         channels=MPL5_specindex_channels,
                         formats={'string': 'Spectral index'},
                         description='Measurements of spectral indices.'),
    MultiChannelProperty('specindex_corr', ivar=False, mask=False,
                         channels=MPL5_specindex_channels,
                         formats={'string': 'Spectral index sigma correction',
                                  'latex': r'Spectral index $\sigma$ correction'},
                         description='Velocity dispersion corrections for the '
                                     'spectral index measurements '
                                     '(can be ignored for D4000, Dn4000).')
]


MPL5_models = [
    Model('binned_flux', 'FLUX', 'WAVE', extension_ivar='IVAR',
          extension_mask='MASK', unit=u.erg / u.s / (u.cm ** 2) / spaxel,
          scale=1e-17, formats={'string': 'Binned flux'},
          description='Flux of the binned spectra',
          binid=binid_property),
    Model('full_fit', 'MODEL', 'WAVE', extension_ivar=None,
          extension_mask='MASK', unit=u.erg / u.s / (u.cm ** 2) / spaxel,
          scale=1e-17, formats={'string': 'Best fitting model'},
          description='The best fitting model spectra (sum of the fitted '
                      'continuum and emission-line models)',
          binid=binid_property),
    Model('emline_fit', 'EMLINE', 'WAVE', extension_ivar=None,
          extension_mask='EMLINE_MASK',
          unit=u.erg / u.s / (u.cm ** 2) / spaxel,
          scale=1e-17, formats={'string': 'Emission line model spectrum'},
          description='The model spectrum with only the emission lines.',
          binid=binid_property),
    Model('emline_base_fit', 'EMLINE_BASE', 'WAVE', extension_ivar=None,
          extension_mask='EMLINE_MASK',
          unit=u.erg / u.s / (u.cm ** 2) / spaxel,
          scale=1e-17, formats={'string': 'Emission line baseline fit'},
          description='The model of the constant baseline fitted beneath the '
                      'emission lines.',
          binid=binid_property)
]

MPL5 = DAPDataModel('2.0.2', aliases=['MPL-5', 'MPL5'], bintypes=[ALL, NRE, VOR10, SPX],
                    templates=[GAU_MILESHC], properties=MPL5_maps, bitmasks=get_maskbits('MPL-5'),
                    models=MPL5_models, default_bintype='SPX', default_template='GAU-MILESHC',
                    property_table='SpaxelProp5', default_binid=binid_property,
                    default_mapmask=['NOCOV', 'UNRELIABLE', 'DONOTUSE'], qual_flag='DAPQUAL')
