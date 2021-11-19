# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-04-04 23:07:41
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-10-18 14:12:38

from __future__ import absolute_import, division, print_function

import copy

from marvin.utils.datamodel.maskbit import get_maskbits

from .base import DAPDataModel
from .MPL6 import GAU_MILESHC, HYB10, VOR10, MPL6_maps, MPL6_models, binid_properties


# MPL-7 DapDataModel goes here
MPL7 = DAPDataModel('2.2.1', aliases=['MPL-7', 'MPL7', 'DR15', 'DR16'],
                    bintypes=[HYB10, VOR10],
                    db_only=[HYB10],
                    templates=[GAU_MILESHC],
                    properties=MPL6_maps,
                    models=MPL6_models,
                    bitmasks=get_maskbits('MPL-7'),
                    default_bintype='HYB10',
                    default_template='GAU-MILESHC',
                    property_table='SpaxelProp7',
                    default_binid=copy.deepcopy(binid_properties[0]),
                    default_mapmask=['NOCOV', 'UNRELIABLE', 'DONOTUSE'],
                    qual_flag='DAPQUAL')
