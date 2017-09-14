# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-22 22:43:15
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-13 15:59:46

from __future__ import print_function, division, absolute_import

from .base import *

from .MPL4 import MPL4
from .MPL5 import MPL5


# Defines the list of datamodels.
datamodel = QueryDataModelList([MPL4, MPL5])
