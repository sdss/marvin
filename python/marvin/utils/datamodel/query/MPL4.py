# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-20 13:24:13
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-26 10:06:47

from __future__ import print_function, division, absolute_import

from marvin.utils.datamodel.query.base import query_params
from .base import QueryDataModel
import copy as copy_mod

GROUPS = copy_mod.deepcopy(query_params)

# list of tables to exclude
BASE_EXCLUDE = ['anime', 'catalogue', 'pipeline', 'maskbit', 'hdu',
                'extcol', 'exttype', 'extname', 'cube_shape', 'spaxelprops']

EXCLUDE = ['modelcube', 'modelspaxel', 'redcorr', 'obsinfo', 'dapall'] + BASE_EXCLUDE

MPL4 = QueryDataModel(release='MPL-4', groups=GROUPS, aliases=['MPL4', 'v1_5_1', '1.1.1'], exclude=EXCLUDE)

GRPDICT = {'Emission': 'spaxelprop.emline', 'Kinematic': 'spaxelprop.stellar', 'Spectral Indices': 'spaxelprop.specindex', 'NSA Catalog': 'nsa'}
MPL4.regroup(GRPDICT)

#MPL4.add_group('Other')

