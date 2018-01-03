# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-22 22:43:15
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-29 11:09:58

from __future__ import print_function, division, absolute_import

from .base import QueryDataModelList

from .MPL4 import MPL4
from .MPL5 import MPL5
from .MPL6 import MPL6

# Defines the list of datamodels.
datamodel = QueryDataModelList([MPL4, MPL5, MPL6])


