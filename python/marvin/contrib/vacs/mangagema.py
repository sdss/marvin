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

        # get any parameters you need from the parent object
        mangaid = parent_object.mangaid

        # define the variables to build a unique path to your VAC file
        path_params = {'ver': self.version[release]}

        # get_path returns False if the files do not exist locally
        allfile = self.get_path('mangagema', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not allfile:
            allfile = self.download_vac('mangagema', path_params=path_params)

        # create container for more complex return data
        gemadata = GEMAData(allfile=allfile)

        return gemadata

class HIData(object):
    ''' A customized class to handle more complex data

    This class handles data from both the HI summary file and the
    individual HI spectral files.  Row data from the summary file
    is returned via the `data` property.  Spectral data can be plotted via
    the `plot_spectrum` method.

    '''

    def __init__(self, plateifu, allfile=None, specfile=None):
        self._allfile = allfile
        self._specfile = specfile
        self._plateifu = plateifu
        self._hi_data = self._open_file(allfile)
        self._indata = plateifu in self._hi_data['plateifu']
        self._specdata = None

    def __repr__(self):
        return 'HI({0})'.format(self._plateifu)

    @staticmethod
    def _open_file(hifile):
        return astropy.io.fits.getdata(hifile, 1)

    @property
    def data(self):
        ''' Returns the FITS row data from the mangaHIall summary file '''

        if not self._indata:
            return "No HI data exists for {0}".format(self._plateifu)

        idx = self._hi_data['plateifu'] == self._plateifu
        return self._hi_data[idx]







