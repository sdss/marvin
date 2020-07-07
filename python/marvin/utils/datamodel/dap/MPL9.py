# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Filename: MPL9.py
# Project: dap
# Author: Brian Cherinka
# Created: Monday, 4th November 2019 2:32:54 pm
# License: BSD 3-clause "New" or "Revised" License
# Copyright (c) 2019 Brian Cherinka
# Last Modified: Monday, 11th November 2019 3:56:19 pm
# Modified By: Brian Cherinka


from __future__ import print_function, division, absolute_import

import copy
from astropy import units as u
from marvin.utils.datamodel.maskbit import get_maskbits
from .base import DAPDataModel, Template, spaxel as spaxel_unit, reindex_channels
from .base import Model, Channel, MultiChannelProperty
from .MPL6 import SPX, HYB10, VOR10, MPL6_maps, binid_properties
from .MPL6 import MPL6_emline_channels, oii_channel, oiid_channel

from .MPL6 import MPL6_specindex_channels
from .MPL8 import SPX, HYB10, VOR10, MPL8_maps
from .MPL8 import MPL8_models, MPL8_emline_channels

# update Template for MPL-8
MILESHC_MASTARHC = Template('MILESHC-MASTARHC',
                            description=('Stellar kinematic templates from the MILES library.  '
                                         'Stellar continuum template derived from high-res spectra '
                                         'from the MaStar stellary library, used during emission-line fits.'))

emline_channels = copy.deepcopy(MPL8_emline_channels)
MPL8_specindex_channels = copy.deepcopy(MPL6_specindex_channels)
updated_maps = copy.deepcopy(MPL8_maps)

# new coo channels
coo_channel = Channel('r_h_kpc', formats={'string': 'R/(h/kpc)'}, idx=2)
for t in updated_maps:
    if t.name in ['spx_ellcoo', 'bin_lwellcoo']:
        t.append_channel(coo_channel, at_index=2, unit=u.kpc / u.h)


# new emission line channels
new_channels = [
    Channel('ariii_7137', formats={'string': 'ArIII 7137',
                                   'latex': r'$\forb{Ar\,III}\;\lambda 7137$'}, idx=28),
    Channel('ariii_7753', formats={'string': 'ArIII 7753',
                                   'latex': r'$\forb{Ar\,III}\;\lambda 7753$'}, idx=29),
    Channel('hei_7067', formats={'string': 'HeI 7067',
                                 'latex': r'$\forb{He\,I}\;\lambda 7067$'}, idx=27),
    Channel('h11_3771', formats={'string': 'H11 3771',
                                 'latex': r'$\forb{H\,11}\;\lambda 3771$'}, idx=3),
    Channel('h12_3751', formats={'string': 'H12 3751',
                                 'latex': r'$\forb{H\,12}\;\lambda 3751$'}, idx=2),
    Channel('siii_9071', formats={'string': 'SIII 9071',
                                 'latex': r'$\forb{S\,III}\;\lambda 9071$'}, idx=31),
    Channel('siii_9533', formats={'string': 'SIII 9533',
                                 'latex': r'$\forb{S\,III}\;\lambda 9533$'}, idx=33),
    Channel('peps_9548', formats={'string': 'P-epsilon 9548',
                                  'latex': r'P$\epsilon\;\lambda 9548$'}, idx=34),
    Channel('peta_9017', formats={'string': 'P-eta 9017',
                                  'latex': r'P$\eta\;\lambda 9017$'}, idx=30),
    Channel('pzet_9231', formats={'string': 'P-zeta 9231',
                                  'latex': r'P$\zeta\;\lambda 9231$'}, idx=32),
]

# create new MPL9 channels and reindex
tmp = emline_channels + new_channels
idx_list = ['oii_3729', 'h12_3751', 'h11_3771', 'hthe_3798', 'heta_3836', 'neiii_3869', 'hei_3889',
            'hzet_3890', 'neiii_3968', 'heps_3971', 'hdel_4102', 'hgam_4341', 'heii_4687', 'hb_4862',
            'oiii_4960', 'oiii_5008', 'ni_5199', 'ni_5201', 'hei_5877', 'oi_6302', 'oi_6365',
            'nii_6549', 'ha_6564', 'nii_6585', 'sii_6718', 'sii_6732', 'hei_7067', 'ariii_7137',
            'ariii_7753', 'peta_9017', 'siii_9071', 'pzet_9231', 'siii_9533', 'peps_9548']
MPL9_emline_channels = reindex_channels(tmp, names=idx_list, starting_idx=1)

# update existing emline properties with new channel list
for m in updated_maps:
    if 'emline' in m.name and '_fom' not in m.name:
        unit, scale = m[0].unit, m[0].unit.scale
        params = {'ivar': m[0].ivar, 'mask': m[0].mask, 'formats': m[0].formats,
                  'pixmask_flag': m[0].pixmask_flag}
        scale = unit.scale
        oii = oiid_channel if 'd' in m.channels[0].name else oii_channel
        chans = [oii] + MPL9_emline_channels
        m.update_channels(chans, unit=unit, scale=scale, **params)


# new properties
new_properties = [
    # New Emission Line extensions
    MultiChannelProperty('emline_sew_cnt', ivar=False, mask=False,
                         channels=[oiid_channel] + MPL9_emline_channels,
                         formats={'string': 'Continuum of Summed EW'},
                         unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit, scale=1e-17,
                         binid=binid_properties[3],
                         description='Continuum used for summed-flux equivalent width measurement'),
    MultiChannelProperty('emline_gew_cnt', ivar=False, mask=False,
                         channels=[oii_channel] + MPL9_emline_channels,
                         formats={'string': 'Continuum of Gaussian-fit EW'},
                         unit=u.erg / u.s / (u.cm ** 2) / u.Angstrom / spaxel_unit, scale=1e-17,
                         binid=binid_properties[3],
                         description='Continuum used for Gaussian-fit equivalent width measurement'),
    # New Spectral Index extensions
    MultiChannelProperty('specindex_bcen', ivar=False, mask=False,
                         channels=MPL8_specindex_channels,
                         formats={'string': 'Flux-weighted center: blue sideband',
                                  'latex': r'Flux-weighted center: blue sideband'},
                         description='Flux-weighted center of the blue sideband in '
                                     'the spectral-index measurement'),
    MultiChannelProperty('specindex_bcnt', ivar=False, mask=False,
                         channels=MPL8_specindex_channels,
                         formats={'string': 'Continuum: blue sideband',
                                  'latex': r'Continuum: blue sideband'},
                         description='Continuum level in the blue sideband used in '
                                     'the spectral-index measurement'),
    MultiChannelProperty('specindex_rcen', ivar=False, mask=False,
                         channels=MPL8_specindex_channels,
                         formats={'string': 'Flux-weighted center: red sideband',
                                  'latex': r'Flux-weighted center: red sideband'},
                         description='Flux-weighted center of the red sideband in '
                                     'the spectral-index measurement'),
    MultiChannelProperty('specindex_rcnt', ivar=False, mask=False,
                         channels=MPL8_specindex_channels,
                         formats={'string': 'Continuum: red sideband',
                                  'latex': r'Continuum: red sideband'},
                         description='Continuum level in the red sideband used in '
                                     'the spectral-index measurement'),
    MultiChannelProperty('specindex_model', ivar=False, mask=False,
                         channels=MPL8_specindex_channels,
                         formats={'string': 'Best-fit Index Measurement',
                                  'latex': r'Best-fit Index Measurement'},
                         description='Index measurement made using the best-fitting '
                                     'stellar continuum model')
]

MPL9_maps = updated_maps + new_properties

# MPL-9 DapDataModel goes here
MPL9 = DAPDataModel('2.4.1', aliases=['MPL-9', 'MPL9'],
                    bintypes=[SPX, HYB10, VOR10],
                    templates=[MILESHC_MASTARHC],
                    properties=MPL9_maps,
                    models=MPL8_models,
                    bitmasks=get_maskbits('MPL-9'),
                    default_bintype='HYB10',
                    default_template='MILESHC-MASTARHC',
                    property_table='SpaxelProp9',
                    default_binid=copy.deepcopy(binid_properties[0]),
                    default_mapmask=['NOCOV', 'UNRELIABLE', 'DONOTUSE'],
                    qual_flag='DAPQUAL')
