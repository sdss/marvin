# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-07-17 23:36:37
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-19 15:44:42

from __future__ import print_function, division, absolute_import

from collections import defaultdict
from marvin.utils.datamodel.drp import datamodel
from marvin.contrib.vacs.base import VACMixIn
from .base import VACList, VACDataModel


subvacs = VACMixIn.__subclasses__()
vacdms = []

# create a dictionary of VACs by release
vacdict = defaultdict(list)
for sv in subvacs:
    # skip hidden VACs
    if sv._hidden:
        continue
    # add versions to dictionary
    for k in sv.version.keys():
        vacdict[k].append(sv)

# create VAC datamodels
for release, vacs in vacdict.items():
    vc = VACList(vacs)
    dm = datamodel[release] if release in datamodel else None
    aliases = dm.aliases if dm else None
    vacdm = VACDataModel(release, vacs=vc, aliases=aliases)
    vacdms.append(vacdm)
