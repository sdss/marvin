# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-20 14:20:06
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-26 10:50:55

from __future__ import print_function, division, absolute_import
from .base import QueryDataModel
import copy
from marvin.utils.datamodel.query.base import query_params
from marvin.utils.datamodel.query.MPL4 import EXCLUDE

GROUPS = copy.deepcopy(query_params)

# list of tables to exclude
EXCLUDE = set(EXCLUDE) - set(['modelcube', 'modelspaxel', 'redcorr']) | set(['executionplan', 'current_default'])

MPL5 = QueryDataModel(release='MPL-5', groups=GROUPS, aliases=['MPL5', 'v2_0_2', '2.0.1'], exclude=EXCLUDE)

