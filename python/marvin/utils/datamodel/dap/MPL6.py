# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-13 16:05:56
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-13 16:19:15

from __future__ import print_function, division, absolute_import
from .base import Bintype, Template, DAPDataModel, Property, MultiChannelProperty, spaxel, Channel
from .MPL5 import GAU_MILESHC

# MPL5 = DAPDataModel('2.0.2', aliases=['MPL-5', 'MPL5'], bintypes=[ALL, NRE, VOR10, SPX],
#                     templates=[GAU_MILESHC], properties=MPL5_maps,
#                     default_bintype='SPX', default_template='GAU-MILESHC')

# GAU_MILESHC = Template('GAU-MILESHC',
#                        description='Set of stellar templates derived from the MILES library.')

# MPL-6 DapDataModel goes here
MPL6 = DAPDataModel('trunk', aliases=['MPL-6', 'MPL6'],
                    templates=[GAU_MILESHC], default_template='GAU-MILESHC')


