# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-20 13:24:13
# @Last modified by:   andrews
# @Last modified time: 2018-10-16 10:10:91

from __future__ import absolute_import, division, print_function

import copy
from marvin.utils.datamodel.dap import datamodel
from marvin.utils.datamodel.query.base import query_params
from .base import QueryDataModel


def groups():
    return copy.deepcopy(query_params)

# MPL-4

# list of tables to exclude
BASE_EXCLUDE = ['anime', 'catalogue', 'pipeline', 'maskbit', 'hdu', 'query', 'user',
                'extcol', 'exttype', 'extname', 'cube_shape', 'spaxelprops', 'testtable']

EXCLUDE = ['modelcube', 'modelspaxel', 'redcorr', 'obsinfo', 'dapall'] + BASE_EXCLUDE

MPL4 = QueryDataModel(release='MPL-4', groups=groups(), aliases=['MPL4', 'v1_5_1', '1.1.1'], exclude=EXCLUDE, dapdm=datamodel['MPL-4'])


# MPL-5

# list of tables to exclude
EX5 = set(EXCLUDE) - set(['modelcube', 'modelspaxel', 'redcorr']) | set(['executionplan', 'current_default'])

MPL5 = QueryDataModel(release='MPL-5', groups=groups(), aliases=['MPL5', 'v2_0_2', '2.0.1'], exclude=EX5, dapdm=datamodel['MPL-5'])


# MPL-6

# list of tables to exclude
EX6 = set(EX5) - set(['obsinfo', 'dapall'])

MPL6 = QueryDataModel(release='MPL-6', groups=groups(), aliases=['MPL6', 'v2_3_1', '2.1.3'], exclude=EX6, dapdm=datamodel['MPL-6'])

# MPL-7

MPL7 = QueryDataModel(release='MPL-7', groups=groups(), aliases=['MPL7', 'v2_4_3', '2.2.0', 'DR15'], exclude=EX6, dapdm=datamodel['MPL-7'])
