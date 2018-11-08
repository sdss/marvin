# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-07-17 23:33:27
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-18 23:09:17

from __future__ import print_function, division, absolute_import

from marvin.utils.datamodel.vacs.base import VACDataModelList
from marvin.utils.datamodel.vacs.releases import vacdms

# build the full list
datamodel = VACDataModelList(vacdms)
