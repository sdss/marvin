# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Filename: MPL8.py
# Project: dap
# Author: Brian Cherinka
# Created: Friday, 8th February 2019 2:10:30 pm
# License: BSD 3-clause "New" or "Revised" License
# Copyright (c) 2019 Brian Cherinka
# Last Modified: Thursday, 2nd May 2019 12:34:58 pm
# Modified By: Brian Cherinka


from __future__ import print_function, division, absolute_import

import copy
from astropy import units as u
from marvin.utils.datamodel.maskbit import get_maskbits
from .base import DAPDataModel, Template, spaxel as spaxel_unit
from .base import Model, Channel, MultiChannelProperty
from .MPL6 import SPX, HYB10, VOR10, MPL6_maps, binid_properties
from .MPL6 import MPL6_emline_channels, oii_channel, oiid_channel

# update Template for MPL-8
MILESHC_MILESHC = Template('MILESHC-MILESHC',
                           description=('Set of templates derived from the MILES library, '
                                        'used for both the stellar kinematics and emission lines.'))


# new figure of merit channels
fom_channels = [
    Channel('rms', formats={'string': 'RMS of residuals for fitted pixels',
                            'latex': r'RMS of residuals'}, 
            description='RMS of the residuals for all fitted pixels in the stellar continuum.',
            unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit, scale=1e-17, idx=0),
    Channel('frms', formats={'string': 'RMS of the fractional residuals',
                             'latex': r'RMS of the fractional residuals'}, 
            description='Fractional residuals for the stellar continuum fit.', idx=1),
    Channel('rchi2', formats={'string': 'Stellar continuum reduced chi-square',
                              'latex': r'Stellar\ continuum\ reduced\ \chi^2'},
            description='Reduced chi-square of the stellar continuum fit.', idx=2),
    Channel('fresid_68th_percentile', db_name='68th_perc_frac_resid', 
            formats={'string': '68th percentile',
                     'latex': r'68^{th} percentile'}, 
            description='68%% growth of the fractional residuals between the model and data.', idx=3),
    Channel('fresid_99th_percentile', db_name='99th_perc_frac_resid',
            formats={'string': '99th percentile Frac. Residuals',
                     'latex': r'99^{th} percentile Frac. Residuals'},
            description='99%% growth of the fractional residuals between the model and data.', idx=4),
    Channel('fresid_max', db_name='max_frac_resid',
            formats={'string': 'Max Fractional Residual',
                     'latex': r'Max Fractional Residual'},
            description='Maximum growth of the fractional residuals between the model and data.', idx=5),
    Channel('per_pix_chi_68th_percentile', db_name='68th_perc_per_pix_chi',
            formats={'string': '68th percentile',
                     'latex': r'68^{th} percentile'},
            description='68%% growth of the error-normalized residuals', idx=6),
    Channel('per_pix_chi_99th_percentile', db_name='99th_perc_per_pix_chi',
            formats={'string': '99th percentile',
                     'latex': r'99^{th} percentile'},
            description='99%% growth of the error-normalized residuals', idx=7),
    Channel('per_pix_chi_max', db_name='max_per_pix_chi',
            formats={'string': 'Max Error-Normalized Residuals',
                     'latex': r'Max Error-Normalized Residuals'},
            description='Maximum growth of the error-normalized residuals', idx=8),
]

# new emission line channels
new_channels = [
    Channel('hei_3889', formats={'string': 'HeI 3889',
                                 'latex': r'$\forb{He\,I\]\;\lambda 3889$'}, idx=22),
    Channel('ni_5199', formats={'string': 'NI 5199',
                                'latex': r'$\forb{N\,I\]\;\lambda 5199$'}, idx=23),
    Channel('ni_5201', formats={'string': 'NI 5201',
                                'latex': r'$\forb{N\,I\]\;\lambda 5201$'}, idx=24),
]

MPL8_emline_channels = copy.deepcopy(MPL6_emline_channels) + new_channels

new_properties = [
    # New Emission Line extensions
    MultiChannelProperty('emline_ga', ivar=False, mask=False,
                         channels=[oii_channel] + MPL8_emline_channels,
                         formats={'string': 'Amplitude of Fitted Gaussians'},
                         unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit, scale=1e-17,
                         binid=binid_properties[3],
                         description='Amplitude of the fitted Gaussian emission lines.'),
    MultiChannelProperty('emline_ganr', ivar=False, mask=False,
                         channels=[oii_channel] + MPL8_emline_channels,
                         formats={'string': 'Amplitude Over Noise of Fitted Gaussians'},
                         binid=binid_properties[3],
                         description='Amplitude Over Noise of the fitted Gaussian emission lines.'),
    MultiChannelProperty('emline_lfom', ivar=False, mask=False,
                         channels=[oii_channel] + MPL8_emline_channels,
                         formats={'string': 'Emission Line Reduced Chi-Square'},
                         binid=binid_properties[3],
                         description='Reduced chi-square in 15 pixel windows around each fitted emission line'),
    # Figure of Merit Extensions in MAPS
    MultiChannelProperty('emline_fom', ivar=False, mask=False,
                         channels=fom_channels,
                         formats={'string': 'Emission Line FOM'},
                         unit=u.dimensionless_unscaled,
                         description='Full spectrum figures-of-merit for emission line fitting'),
    MultiChannelProperty('stellar_fom', ivar=False, mask=False,
                         channels=fom_channels, formats={
                             'string': 'Stellar FOM'},
                         unit=u.dimensionless_unscaled,
                         description='Full spectrum figures-of-merit for stellar continuum fitting'),
    # New Stellar SigmaCorr
    MultiChannelProperty('stellar_sigmacorr', ivar=False, mask=False,
                         channels=[Channel('resolution_difference',
                                           formats={'string': 'Resolution Difference'}, idx=0),
                                   Channel('fit',
                                           formats={'string': 'Fit'}, idx=1)],
                         unit=u.km / u.s,
                         formats={'string': 'Stellar sigma correction',
                                  'latex': r'Stellar $\sigma$ correction'},
                         description='Quadrature correction for STELLAR_SIGMA to obtain the '
                                     'astrophysical velocity dispersion.'),
]

emline_props = [
    MultiChannelProperty('emline_sflux', ivar=True, mask=True,
                         channels=[oiid_channel] + MPL8_emline_channels,
                         formats={'string': 'Emission line summed flux'},
                         unit=u.erg / u.s / (u.cm ** 2) / spaxel_unit, scale=1e-17,
                         binid=binid_properties[3],
                         description='Non-parametric summed flux for emission lines.'),
    MultiChannelProperty('emline_sew', ivar=True, mask=True,
                         channels=[oiid_channel] + MPL8_emline_channels,
                         formats={'string': 'Emission line EW'},
                         unit=u.Angstrom,
                         binid=binid_properties[3],
                         description='Emission line non-parametric equivalent '
                                     'widths measurements.'),
    MultiChannelProperty('emline_gflux', ivar=True, mask=True,
                         channels=[oii_channel] + MPL8_emline_channels,
                         formats={'string': 'Emission line Gaussian flux'},
                         unit=u.erg / u.s / (u.cm ** 2) / spaxel_unit, scale=1e-17,
                         binid=binid_properties[3],
                         description='Gaussian profile integrated flux for emission lines.'),
    MultiChannelProperty('emline_gvel', ivar=True, mask=True,
                         channels=[oii_channel] + MPL8_emline_channels,
                         formats={'string': 'Emission line Gaussian velocity'},
                         unit=u.km / u.s,
                         binid=binid_properties[3],
                         description='Gaussian profile velocity for emission lines.'),
    MultiChannelProperty('emline_gew', ivar=True, mask=True,
                         channels=[oii_channel] + MPL8_emline_channels,
                         formats={'string': 'Emission line Gaussian EW'},
                         unit=u.Angstrom,
                         binid=binid_properties[3],
                         description='Gaussian-fitted equivalent widths measurements '
                                     '(based on EMLINE_GFLUX).'),
    MultiChannelProperty('emline_gsigma', ivar=True, mask=True,
                         channels=[oii_channel] + MPL8_emline_channels,
                         formats={'string': 'Emission line Gaussian sigma',
                                  'latex': r'Emission line Gaussian $\sigma$'},
                         unit=u.km / u.s,
                         binid=binid_properties[3],
                         description='Gaussian profile velocity dispersion for emission lines; '
                                     'must be corrected using EMLINE_INSTSIGMA.'),
    MultiChannelProperty('emline_instsigma', ivar=False, mask=False,
                         channels=[oii_channel] + MPL8_emline_channels,
                         formats={'string': 'Emission line instrumental sigma',
                                  'latex': r'Emission line instrumental $\sigma$'},
                         unit=u.km / u.s,
                         binid=binid_properties[3],
                         description='Instrumental dispersion at the fitted line center.'),
    MultiChannelProperty('emline_tplsigma', ivar=False, mask=False,
                         channels=[oii_channel] + MPL8_emline_channels,
                         formats={'string': 'Emission line template instrumental sigma',
                                  'latex': r'Emission line template instrumental $\sigma$'},
                         unit=u.km / u.s,
                         binid=binid_properties[3],
                         description='The dispersion of each emission line used in '
                                     'the template spectra'),
]

# Update Maps Properties
remove = ['stellar_cont_rchi2', 'stellar_sigmacorr', 'stellar_cont_fresid']
MPL8_maps = [m for m in MPL6_maps if m.name not in remove and 'emline' not in m.name]
MPL8_maps += emline_props + new_properties

# Update Model Properties
MPL8_models = [
    Model('binned_flux', 'FLUX', 'WAVE', extension_ivar='IVAR',
          extension_mask='MASK', unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit,
          scale=1e-17, formats={'string': 'Binned flux'},
          description='Flux of the binned spectra',
          binid=binid_properties[0]),
    Model('full_fit', 'MODEL', 'WAVE', extension_ivar=None,
          extension_mask='MODEL_MASK', unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit,
          scale=1e-17, formats={'string': 'Best fitting model'},
          description='The best fitting model spectra (sum of the fitted '
                      'continuum and emission-line models)',
          binid=binid_properties[0]),
    Model('emline_fit', 'EMLINE', 'WAVE', extension_ivar=None,
          extension_mask=None,
          unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit,
          scale=1e-17, formats={'string': 'Emission line model spectrum'},
          description='The model spectrum with only the emission lines.',
          binid=binid_properties[3]),
    Model('stellar_fit', 'STELLAR', 'WAVE', extension_ivar=None,
          extension_mask='STELLAR_MASK',
          unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit,
          scale=1e-17, formats={'string': 'Stellar continuum model spectrum'},
          description='The model spectrum with only the stellar continuum',
          binid=binid_properties[1])
]

# MPL-8 DapDataModel goes here
MPL8 = DAPDataModel('2.3.0', aliases=['MPL-8', 'MPL8'],
                    bintypes=[SPX, HYB10, VOR10],
                    templates=[MILESHC_MILESHC],
                    properties=MPL8_maps,
                    models=MPL8_models,
                    bitmasks=get_maskbits('MPL-8'),
                    default_bintype='HYB10',
                    default_template='MILESHC-MILESHC',
                    property_table='SpaxelProp8',
                    default_binid=copy.deepcopy(binid_properties[0]),
                    default_mapmask=['NOCOV', 'UNRELIABLE', 'DONOTUSE'],
                    qual_flag='DAPQUAL')
