# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-20 15:14:40
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-20 15:15:44

from __future__ import print_function, division, absolute_import
from .base import QueryDataModel
from marvin.utils.datamodel.query.MPL5 import GROUPS, EXCLUDE

# list of tables to exclude
EXCLUDE = set(EXCLUDE) - set(['obsinfo', 'dapall'])

MPL6 = QueryDataModel(release='MPL-6', groups=GROUPS, aliases=['MPL6', 'v2_2_0', 'trunk'], exclude=EXCLUDE)

