#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-08-08
# @Filename: MPL5.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-30 11:44:55


from __future__ import absolute_import, division, print_function

from astropy import units as u

from marvin.utils.datamodel.maskbit import get_maskbits

from .base import Bintype, Channel, DAPDataModel, MultiChannelProperty, Property, Template, spaxel


M11_STELIB_ZSOL = Template('M11-STELIB-ZSOL', n=0,
                           description='M11 stellar template, solar metallicity.')
MIUSCAT_THIN = Template('MIUSCAT-THIN', n=1, description='MIUSCAT stellar template.')
MILES_THIN = Template('MILES-THIN', n=2, description='MILES stellar template.')

NONE = Bintype('NONE', binned=False, n=3, description='No binning.')
RADIAL = Bintype('RADIAL', n=7, description='Bin spectra from 0-1 Re and from 1-2 Re in two bins.')
STON = Bintype('STON', n=1, description='Bin to S/N=30; only include S/N>5 spectra; '
                                        'fit V, sigma, h3, h4 for stellar kinematics.')


binid_property = Property('binid', ivar=False, mask=False, channel=None,
                          formats={'string': 'Bin ID'},
                          description='ID number for the bin for which the pixel value was '
                                      'calculated; bins are sorted by S/N.')


MPL4_emline_channels = [
    Channel('oiid_3728', formats={'string': 'OIId 3728',
                                  'latex': r'$\forb{O\,II}\;\lambda\lambda 3726,3728$'}, idx=0,
            description='Single Gaussian fit to the (unresolved) OII 3727 doublet feature'),
    Channel('hb_4862', formats={'string': 'H-beta 4862',
                                'latex': r'H$\beta\;\lambda 4862$'}, idx=1),
    Channel('oiii_4960', formats={'string': 'OIII 4960',
                                  'latex': r'$\forb{O\,III}\;\lambda 4960$'}, idx=2),
    Channel('oiii_5008', formats={'string': 'OIII 5008',
                                  'latex': r'$\forb{O\,III}\;\lambda 5008$'}, idx=3),
    Channel('oi_6302', formats={'string': 'OI 6302',
                                'latex': r'$\forb{O\,I}\;\lambda 6302$'}, idx=4),
    Channel('oi_6365', formats={'string': 'OI 6365',
                                'latex': r'$\forb{O\,I}\;\lambda 6365$'}, idx=5),
    Channel('nii_6549', formats={'string': 'NII 6549',
                                 'latex': r'$\forb{N\,II}\;\lambda 6549$'}, idx=6),
    Channel('ha_6564', formats={'string': 'H-alpha 6564',
                                'latex': r'H$\alpha\;\lambda 6564$'}, idx=7),
    Channel('nii_6585', formats={'string': 'NII 6585',
                                 'latex': r'$\forb{N\,II}\;\lambda 6585$'}, idx=8),
    Channel('sii_6718', formats={'string': 'SII 6718',
                                 'latex': r'$\forb{S\,II}\;\lambda 6718$'}, idx=9),
    Channel('sii_6732', formats={'string': 'SII 6732',
                                 'latex': r'$\forb{S\,II\]\;\lambda 6732$'}, idx=10)
]

MPL4_specindex_channels = [
    Channel('d4000', formats={'string': 'D4000'}, unit=u.dimensionless_unscaled, idx=0),
    Channel('caii0p39', formats={'string': 'CaII 0p39'}, unit=u.Angstrom, idx=1),
    Channel('hdeltaa', formats={'string': 'HDeltaA',
                                'latex': r'H\delta\,A'}, unit=u.Angstrom, idx=2),
    Channel('cn1', formats={'string': 'CN1'}, unit=u.mag, idx=3),
    Channel('cn2', formats={'string': 'CN2'}, unit=u.mag, idx=4),
    Channel('ca4227', formats={'string': 'Ca 4227',
                               'latex': r'Ca\,\lambda 4227'}, unit=u.Angstrom, idx=5),
    Channel('hgammaa', formats={'string': 'HGammaA',
                                'latex': r'H\gamma\,A'}, unit=u.Angstrom, idx=6),
    Channel('fe4668', formats={'string': 'Fe 4668',
                               'latex': r'Fe\,\lambda 4668'}, unit=u.Angstrom, idx=7),
    Channel('hb', formats={'string': 'Hb',
                           'latex': r'H\beta'}, unit=u.Angstrom, idx=8),
    Channel('mgb', formats={'string': 'Mgb'}, unit=u.Angstrom, idx=9),
    Channel('fe5270', formats={'string': 'Fe 5270',
                               'latex': r'Fe\,\lambda 5270'}, unit=u.Angstrom, idx=10),
    Channel('fe5335', formats={'string': 'Fe 5335',
                               'latex': r'Fe\,\lambda 5335'}, unit=u.Angstrom, idx=11),
    Channel('fe5406', formats={'string': 'Fe 5406',
                               'latex': r'Fe\,\lambda 5406'}, unit=u.Angstrom, idx=12),
    Channel('nad', formats={'string': 'NaD'}, unit=u.Angstrom, idx=13),
    Channel('tio1', formats={'string': 'TiO1'}, unit=u.mag, idx=14),
    Channel('tio2', formats={'string': 'TiO2'}, unit=u.mag, idx=15),
    Channel('nai0p82', formats={'string': 'NaI 0p82'}, unit=u.Angstrom, idx=16),
    Channel('caii0p86a', formats={'string': 'CaII 0p86A'}, unit=u.Angstrom, idx=17),
    Channel('caii0p86b', formats={'string': 'CaII 0p86B'}, unit=u.Angstrom, idx=18),
    Channel('caii0p86c', formats={'string': 'CaII 0p86C'}, unit=u.Angstrom, idx=19),
    Channel('mgi0p88', formats={'string': 'MgI 0p88'}, unit=u.Angstrom, idx=20),
    Channel('tio0p89', formats={'string': 'TiO 0p89'}, unit=u.Angstrom, idx=21),
    Channel('feh0p99', formats={'string': 'FeH 0p99'}, unit=u.Angstrom, idx=22)
]


MPL4_maps = [
    MultiChannelProperty('emline_gflux', ivar=True, mask=True, channels=MPL4_emline_channels,
                         unit=u.erg / u.s / (u.cm ** 2) / spaxel, scale=1e-17,
                         formats={'string': 'Emission line Gaussian flux'},
                         description='Fluxes of emission lines based on a single Gaussian fit.'),
    MultiChannelProperty('emline_gvel', ivar=True, mask=True, channels=MPL4_emline_channels,
                         unit=u.km / u.s,
                         formats={'string': 'Emission line Gaussian velocity'},
                         description='Doppler velocity shifts for emission lines relative to '
                                     'the NSA redshift based on a single Gaussian fit.'),
    MultiChannelProperty('emline_gsigma', ivar=True, mask=True, channels=MPL4_emline_channels,
                         unit=u.km / u.s,
                         formats={'string': 'Emission line Gaussian sigma',
                                  'latex': r'Emission line Gaussian $\sigma$'},
                         description='Velocity dispersions of emission lines based on a '
                                     'single Gaussian fit.'),
    MultiChannelProperty('emline_instsigma', ivar=False, mask=False,
                         channels=MPL4_emline_channels,
                         unit=u.km / u.s,
                         formats={'string': 'Emission line instrumental sigma',
                                  'latex': r'Emission line instrumental $\sigma$'},
                         description='Instrumental velocity dispersion at the line centroids '
                                     'for emission lines (based on a single Gaussian fit).'),
    MultiChannelProperty('emline_ew', ivar=True, mask=True, channels=MPL4_emline_channels,
                         unit=u.Angstrom,
                         formats={'string': 'Emission line EW'},
                         description='Equivalent widths for emission lines based on a '
                                     'single Gaussian fit.'),
    MultiChannelProperty('emline_sflux', ivar=True, mask=True, channels=MPL4_emline_channels,
                         unit=u.erg / u.s / (u.cm ** 2) / spaxel, scale=1e-17,
                         formats={'string': 'Emission line summed flux'},
                         description='Fluxes for emission lines based on integrating the '
                                     'flux over a set of passbands.'),
    Property('stellar_vel', ivar=True, mask=True, channel=None,
             unit=u.km / u.s,
             formats={'string': 'Stellar velocity'},
             description='Stellar velocity measurements.'),
    Property('stellar_sigma', ivar=True, mask=True, channel=None,
             unit=u.km / u.s,
             formats={'string': 'Stellar velocity dispersion', 'latex': r'Stellar $\sigma$'},
             description='Stellar velocity dispersion measurements.'),
    MultiChannelProperty('specindex', ivar=True, mask=True,
                         formats={'string': 'Spectral index'},
                         channels=MPL4_specindex_channels,
                         description='Measurements of spectral indices.'),
    binid_property
]


MPL4 = DAPDataModel('1.1.1', aliases=['MPL-4', 'MPL4'], bintypes=[NONE, RADIAL, STON],
                    templates=[M11_STELIB_ZSOL, MIUSCAT_THIN, MILES_THIN],
                    properties=MPL4_maps, bitmasks=get_maskbits('MPL-4'),
                    default_bintype='NONE', default_template='MIUSCAT-THIN',
                    property_table='SpaxelProp', default_binid=binid_property,
                    qual_flag=None)
