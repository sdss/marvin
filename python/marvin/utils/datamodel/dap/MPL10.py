# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Filename: MPL10.py
# Project: dap
# Author: Brian Cherinka
# Created: Monday, 13th July 2020 2:49:16 pm
# License: BSD 3-clause "New" or "Revised" License
# Copyright (c) 2020 Brian Cherinka
# Last Modified: Monday, 13th July 2020 2:49:16 pm
# Modified By: Brian Cherinka


from __future__ import print_function, division, absolute_import
import copy
from astropy import units as u
from marvin.utils.datamodel.maskbit import get_maskbits
from .base import DAPDataModel, Template
from .base import Model, MultiChannelProperty

from .MPL8 import SPX, HYB10, VOR10, binid_properties, MPL8_models
from .MPL9 import MPL9_maps, MPL8_specindex_channels

# update Template for MPL-10
MILESHC_MASTARHC2 = Template('MILESHC-MASTARHC2',
                             description=('Stellar kinematic templates from the MILES library.  '
                                          'Stellar continuum template derived from a hierarchically '
                                          'clustered set of MaStar templates; used during emission-line fits.'))

# MPL-10 maps
updated_maps = copy.deepcopy(MPL9_maps)
remove = ['specindex_bcen', 'specindex_bcnt', 'specindex_rcen', 'specindex_rcnt', 'emline_tplsigma']
updated_maps = [m for m in updated_maps if m.name not in remove]

# new properties
MPL10_specindex_channels = copy.deepcopy(MPL8_specindex_channels)

new_properties = [
    MultiChannelProperty('specindex_bf', ivar=True, mask=True,
                         channels=MPL10_specindex_channels,
                         formats={'string': 'modified BF spectral index'},
                         description='Measurements of spectral indices using a modified definition '
                                     'from Burstein (1984) and Faber (1985).'),
    MultiChannelProperty('specindex_bf_corr', ivar=False, mask=False,
                         channels=MPL10_specindex_channels,
                         formats={'string': 'BF spectral index sigma correction',
                                  'latex': r'BF spectral index $\sigma$ correction'},
                         description='Stellar velocity dispersion corrections for the '
                         'BF spectral index measurements '),
    MultiChannelProperty('specindex_bf_model', ivar=False, mask=False,
                         channels=MPL10_specindex_channels,
                         formats={'string': 'Best-fit BF Index Measurement',
                                  'latex': r'Best-fit BF Index Measurement'},
                         description='BF index measurements made using the best-fitting '
                                     'model spectrum'),
    MultiChannelProperty('specindex_wgt', ivar=True, mask=True,
                         channels=MPL10_specindex_channels,
                         formats={'string': 'Spectral index weights'},
                         description='Index weights to use when calculating an aggregated '
                                     'index for many spaxels/bins.'),
    MultiChannelProperty('specindex_wgt_corr', ivar=False, mask=False,
                         channels=MPL10_specindex_channels,
                         formats={'string': 'Index weight sigma correction',
                                  'latex': r'Index weight $\sigma$ correction'},
                         description='Stellar velocity dispersion corrections for the index weights.'),
    MultiChannelProperty('specindex_wgt_model', ivar=False, mask=False,
                         channels=MPL10_specindex_channels,
                         formats={'string': 'Best-fit Index Weights',
                                  'latex': r'Best-fit Index Weights'},
                         description='The index weights measured on the best-fitting model spectrum.'),
]

MPL10_maps = updated_maps + new_properties

# MPL-10 models
lsf = Model('lsf', 'LSF', 'WAVE', extension_ivar=None,
            extension_mask=None,
            unit=u.Angstrom,
            scale=1, formats={'string': 'Dispersion pre-pixel'},
            description='Broadened pre-pixel dispersion solution (1sigma LSF)',
            binid=binid_properties[0])
MPL10_models = copy.deepcopy(MPL8_models)
MPL10_models.append(lsf)


# MPL-10 DapDataModel goes here
MPL10 = DAPDataModel('3.0.1', aliases=['MPL-10', 'MPL10'],
                     bintypes=[SPX, HYB10, VOR10],
                     db_only=[HYB10],
                     templates=[MILESHC_MASTARHC2],
                     properties=MPL10_maps,
                     models=MPL10_models,
                     bitmasks=get_maskbits('MPL-10'),
                     default_bintype='HYB10',
                     default_template='MILESHC-MASTARHC2',
                     property_table='SpaxelProp10',
                     default_binid=copy.deepcopy(binid_properties[0]),
                     default_mapmask=['NOCOV', 'UNRELIABLE', 'DONOTUSE'],
                     qual_flag='DAPQUAL')
