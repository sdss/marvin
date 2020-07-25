# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-20 13:24:13
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-15 12:00:27

from __future__ import absolute_import, division, print_function

import copy
from marvin import config
from marvin.utils.datamodel.dap import datamodel
from marvin.utils.datamodel.query.base import query_params
from .base import QueryDataModel


# Current Total Parameter Count (with spaxel properties and without)
# all, no spaxelprop, no daptables
# MPL-4 - 571 (309) (309)
# MPL-5 - 703 (322) (301)
# MPL-6 - 1676 (1008) (1031)
# MPL-7 - 1676 (1008) (1031)
# DR15 - 1676 (1008) (1031)
# MPL-8 - xx (xx) (1031)

def groups():
    return copy.deepcopy(query_params)

# exclude the DAP tables
daptables = ['modelcube', 'modelspaxel', 'redcorr']
if not config._allow_DAP_queries:
    # add the spaxelprop table to list of things to remove
    daptables.append('spaxelprop')

# MPL-4

# list of tables to exclude
BASE_EXCLUDE = ['anime', 'catalogue', 'pipeline', 'maskbit', 'hdu', 'query', 'user',
                'extcol', 'exttype', 'extname', 'cube_shape', 'spaxelprops', 'testtable']

EXCLUDE = daptables + ['obsinfo', 'dapall'] + BASE_EXCLUDE

MPL4 = QueryDataModel(release='MPL-4', groups=groups(), aliases=['MPL4', 'v1_5_1', '1.1.1'], exclude=EXCLUDE, dapdm=datamodel['MPL-4'])


# MPL-5

# list of tables to exclude
if not config._allow_DAP_queries:
    # remove the spaxelprop table from the list to re-include
    daptables.remove('spaxelprop')
dapset = set(daptables) if config._allow_DAP_queries else set()
EX5 = set(EXCLUDE) - dapset | set(['executionplan', 'current_default'])

MPL5 = QueryDataModel(release='MPL-5', groups=groups(), aliases=['MPL5', 'v2_0_2', '2.0.1'], exclude=EX5, dapdm=datamodel['MPL-5'])


# MPL-6

# list of tables to exclude
EX6 = set(EX5) - set(['obsinfo', 'dapall'])

MPL6 = QueryDataModel(release='MPL-6', groups=groups(), aliases=['MPL6', 'v2_3_1', '2.1.3'], exclude=EX6, dapdm=datamodel['MPL-6'])

# MPL-7

MPL7 = QueryDataModel(release='MPL-7', groups=groups(), aliases=['MPL7', 'v2_4_3', '2.2.0'], exclude=EX6, dapdm=datamodel['MPL-7'])

# DR15

DR15 = QueryDataModel(release='DR15', groups=groups(), aliases=['DR15', 'v2_4_3', '2.2.0'], exclude=EX6, dapdm=datamodel['DR15'])

# MPL-8

MPL8 = QueryDataModel(release='MPL-8', groups=groups(),
                      aliases=['MPL8', 'v2_5_3', '2.3.0'], exclude=EX6, dapdm=datamodel['MPL-8'])

# DR16

DR16 = QueryDataModel(release='DR16', groups=groups(), aliases=['DR15', 'v2_4_3', '2.2.0'], exclude=EX6, dapdm=datamodel['DR16'])

# MPL-9

MPL9 = QueryDataModel(release='MPL-9', groups=groups(),
                      aliases=['MPL9', 'v2_7_1', '2.4.1'], exclude=EX6, dapdm=datamodel['MPL-9'])

# MPL-10

MPL10 = QueryDataModel(release='MPL-10', groups=groups(),
                       aliases=['MPL10', 'v3_0_1', '3.0.1'], exclude=EX6, dapdm=datamodel['MPL-10'])
