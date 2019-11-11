# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-22 22:43:15
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-15 11:08:16

from __future__ import print_function, division, absolute_import

from .base import QueryDataModelList
from .MPL import MPL4, MPL5, MPL6, MPL7, DR15, MPL8, DR16, MPL9

mpllist = [MPL4, MPL5, MPL6, MPL7, DR15, MPL8, DR16, MPL9]

# Defines the list of datamodels.
datamodel = QueryDataModelList(mpllist)

# Group the datamodel properties for each release
GRPDICT = {'Emission': 'spaxelprop.emline', 'Kinematic': 'spaxelprop.stellar', 'Spectral Indices': 'spaxelprop.specindex',
           'NSA Catalog': 'nsa.'}

for mpl in mpllist:
    # if mpl.release not in config._allowed_releases:
    #     continue

    mpl.regroup(GRPDICT)
    # add header meta
    mpl.add_to_group('Metadata', value='cube_header_keyword.label')
    mpl.add_to_group('Metadata', value='cube_header_value.value')
    mpl.add_to_group('Metadata', value='maps_header_keyword.name')
    mpl.add_to_group('Metadata', value='maps_header_value.value')
    # add the obsinfo group
    mpl.add_group('ObsInfo')
    mpl.add_to_group('ObsInfo', value='obsinfo.')
    # add the dapall summary file group
    if mpl.release not in ['MPL-4', 'MPL-5']:
        mpl.add_group('DAPall Summary')
        mpl.add_to_group('DAPall Summary', value='dapall.')
    # add an misc. group
    mpl.add_group('Other')
    mpl.add_to_group('Other')
