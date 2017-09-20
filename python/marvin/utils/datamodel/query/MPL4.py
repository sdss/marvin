# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-20 13:24:13
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-20 14:39:55

from __future__ import print_function, division, absolute_import

from marvin.tools.query_utils import query_params
from .base import QueryDataModel

GROUPS = query_params.list_groups()

# list of tables to exclude
EXCLUDE = ['modelcube', 'modelspaxel', 'redcorr', 'obsinfo', 'dapall']

MPL4 = QueryDataModel(release='MPL-4', groups=GROUPS, aliases=['MPL4', 'v1_5_1', '1.1.1'], exclude=EXCLUDE)

