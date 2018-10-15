# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-22 22:43:15
# @Last modified by:   andrews
# @Last modified time: 2018-10-16 10:10:16

from __future__ import print_function, division, absolute_import

from .base import QueryDataModelList
from .MPL import MPL4, MPL5, MPL6, MPL7

mpllist = [MPL4, MPL5, MPL6, MPL7]

# Defines the list of datamodels.
datamodel = QueryDataModelList(mpllist)

# Group the datamodel properties for each release
GRPDICT = {'Emission': 'spaxelprop.emline', 'Kinematic': 'spaxelprop.stellar', 'Spectral Indices': 'spaxelprop.specindex', 'NSA Catalog': 'nsa'}

for mpl in mpllist:
    mpl.regroup(GRPDICT)
    mpl.add_group('Other')
    mpl.add_to_group('Other')
