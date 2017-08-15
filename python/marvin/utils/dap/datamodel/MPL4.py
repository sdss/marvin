#!/usr/bin/env python
# encoding: utf-8
#
# MPL4.py
#
# Created by José Sánchez-Gallego on 8 Aug 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from astropy import units as u

from .base import Bintype, Template, DAPDataModel, Property, MultiChannelProperty, spaxel


M11_STELIB_ZSOL = Template('M11-STELIB-ZSOL', n=0)
MIUSCAT_THIN = Template('MIUSCAT-THIN', n=1)
MILES_THIN = Template('MILES-THIN', n=2)

NONE = Bintype('NONE', binned=False, n=3)
RADIAL = Bintype('RADIAL', n=7)
STON = Bintype('STON', n=1)


MPL4_emline_channels = ['oiid_3728', 'hb_4862', 'oiii_4960', 'oiii_5008', 'oi_6302',
                        'oi_6365', 'nii_6549', 'ha_6564', 'nii_6585', 'sii_6718', 'sii_6732']

MPL4_specindex_channels = ['d4000', 'caii0p39', 'hdeltaa', 'cn1', 'cn2', 'ca4227', 'hgammaa',
                           'fe4668', 'hb', 'mgb', 'fe5270', 'fe5335', 'fe5406', 'nad', 'tio1',
                           'tio2', 'nai0p82', 'caii0p86a', 'caii0p86b', 'caii0p86c', 'mgi0p88',
                           'tio0p89', 'feh0p99']

MPL4_specindex_units = [u.Angstrom, u.Angstrom, u.Angstrom, u.mag, u.mag,
                        u.Angstrom, u.Angstrom, u.Angstrom, u.Angstrom,
                        u.Angstrom, u.Angstrom, u.Angstrom, u.Angstrom,
                        u.Angstrom, u.mag, u.mag, u.Angstrom, u.Angstrom,
                        u.Angstrom, u.Angstrom, u.Angstrom, u.Angstrom,
                        u.Angstrom]

MPL4_maps = [
    MultiChannelProperty('emline_gflux', ivar=True, mask=True, channels=MPL4_emline_channels,
                         units=u.erg / u.s / (u.cm ** 2) / spaxel, scales=1e-17,
                         description='Fluxes of emission lines based on a single Gaussian fit.'),
    MultiChannelProperty('emline_gvel', ivar=True, mask=True, channels=MPL4_emline_channels,
                         units=u.km / u.s,
                         description='Doppler velocity shifts for emission lines relative to '
                                     'the NSA redshift based on a single Gaussian fit.'),
    MultiChannelProperty('emline_gsigma', ivar=True, mask=True, channels=MPL4_emline_channels,
                         units=u.km / u.s,
                         description='Velocity dispersions of emission lines based on a '
                                     'single Gaussian fit.'),
    MultiChannelProperty('emline_instsigma', ivar=False, mask=False,
                         channels=MPL4_emline_channels,
                         units=u.km / u.s,
                         description='Instrumental velocity dispersion at the line centroids '
                                     'for emission lines (based on a single Gaussian fit.'),
    MultiChannelProperty('emline_ew', ivar=True, mask=True, channels=MPL4_emline_channels,
                         units=u.Angstrom,
                         description='Equivalent widths for emission lines based on a '
                                     'single Gaussian fit.'),
    MultiChannelProperty('emline_sflux', ivar=True, mask=True, channels=MPL4_emline_channels,
                         units=u.erg / u.s / (u.cm ** 2) / spaxel, scales=1e-17,
                         description='Fluxes for emission lines based on integrating the '
                                     'flux over a set of passbands.'),
    Property('stellar_vel', ivar=True, mask=True, channel=None,
             unit=u.km / u.s,
             description='Stellar velocity measurements.'),
    Property('stellar_sigma', ivar=True, mask=True, channel=None,
             unit=u.km / u.s,
             description='Stellar velocity dispersion measurements.'),
    MultiChannelProperty('specindex', ivar=True, mask=True,
                         channels=MPL4_specindex_channels,
                         units=None,
                         description='Measurements of spectral indices.'),
    Property('binid', ivar=False, mask=False, channel=None,
             unit=None,
             description='ID number for the bin for which the pixel value was '
                         'calculated; bins are sorted by S/N.')
]


MPL4 = DAPDataModel('1.1.1', aliases=['MPL-4'], bintypes=[NONE, RADIAL, STON],
                    templates=[M11_STELIB_ZSOL, MIUSCAT_THIN, MILES_THIN],
                    properties=MPL4_maps,
                    default_bintype='NONE', default_template='MIUSCAT-THIN')
