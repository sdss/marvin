# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Filename: MPL11.py
# Project: dap
# Author: Brian Cherinka
# Created: Friday, 26th February 2021 9:23:41 am
# License: BSD 3-clause "New" or "Revised" License
# Copyright (c) 2021 Brian Cherinka
# Last Modified: Friday, 26th February 2021 9:23:42 am
# Modified By: Brian Cherinka


from __future__ import print_function, division, absolute_import

import copy

from .base import DAPDataModel, Template
from .MPL10 import MILESHC_MASTARHC2, SPX, HYB10, VOR10, MPL10_maps, MPL10_models, binid_properties
from marvin.utils.datamodel.maskbit import get_maskbits

# update Template for MPL-11
MILESHC_MASTARSSP = Template('MILESHC-MASTARSSP',
                             description=('Stellar kinematic templates from the MILES library.  '
                                          'Stellar continuum template derived from a subset of the'
                                          'MaStar Simple Stellar Population models; '
                                          'used during emission-line fits.'))


MPL11_maps = copy.deepcopy(MPL10_maps)
MPL11_models = copy.deepcopy(MPL10_models)


# MPL-11 DapDataModel goes here
MPL11 = DAPDataModel('3.1.0', aliases=['MPL-11', 'MPL11'],
                     bintypes=[SPX, HYB10, VOR10],
                     db_only=[HYB10],
                     templates=[MILESHC_MASTARHC2, MILESHC_MASTARSSP],
                     properties=MPL11_maps,
                     models=MPL11_models,
                     bitmasks=get_maskbits('MPL-11'),
                     default_bintype='HYB10',
                     default_template='MILESHC-MASTARSSP',
                     property_table='SpaxelProp11',
                     default_binid=copy.deepcopy(binid_properties[0]),
                     default_mapmask=['NOCOV', 'UNRELIABLE', 'DONOTUSE'],
                     qual_flag='DAPQUAL')
