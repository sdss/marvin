# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-10-11 17:51:43
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-29 17:23:15

from __future__ import print_function, division, absolute_import

import numpy as np
import astropy
import astropy.units as u
import marvin.tools
from marvin.utils.general.general import get_drpall_table
from marvin.utils.plot.scatter import plot as scatplot

from .base import VACMixIn


class GEMAVAC(VACMixIn):
    """Provides access to the MaNGA-GEMA VAC.

    VAC name: GEMA

    URL: link

    Description: Returns GEMA VAC

    Authors: Maria Argudo-Fernandez, Daniel Goddard, Daniel Thomas, Zheng Zheng, Lihwai Lin, Ting Xiao, Fangting Yuan, Jianhui Lian, et al

    """

    # Required parameters
    name = 'mangagema'
    description = 'Returns GEMA table data'
    version = {'MPL-7': '1_0_1',
               'DR15': '1_0_1'}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube,
               marvin.tools.maps.Maps,
               marvin.tools.modelcube.ModelCube)

    # Required method
    def get_data(self, parent_object):

        # define the variables to build a unique path to your VAC file
        path_params = {'ver': self.version[release]}

        # get_path returns False if the files do not exist locally
        allfile = self.get_path('mangagema', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not allfile:
            allfile = self.download_vac('mangagema', path_params=path_params)

        return allfile









