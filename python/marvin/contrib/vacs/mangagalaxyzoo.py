# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Modified from mangahi.py by Karen Masters. 

from __future__ import print_function, division, absolute_import

import numpy as np
import astropy
import astropy.units as u
import marvin.tools
from marvin.tools.quantities.spectrum import Spectrum

from .base import VACMixIn


class GZVAC(VACMixIn):
    """Provides access to the MaNGA Galaxy Zoo Morphology VAC.

    VAC name: GZ

    URL: link

    Description: Returns Galaxy Zoo morphology for MaNGA galaxies

    Authors: Coleman Krawczyk, Karen Masters and the Galaxy Zoo Team.

    """

    # Required parameters
    name = 'mangagalaxyzoo'
    description = 'Returns Galaxy Zoo morphology'
    version = {'MPL-7': 'v1_0_1','MPL-8': 'v1_0_1',
               'DR15': 'v1_0_1','DR16': 'v1_0_1'}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube,
               marvin.tools.maps.Maps,
               marvin.tools.modelcube.ModelCube)

    # Required method
    def get_data(self, parent_object):

        # get any parameters you need from the parent object
        plateifu = parent_object.plateifu
        release = parent_object.release

        # define the variables to build a unique path to your VAC file
        path_params = {'ver': self.version[release]}

        # get_path returns False if the files do not exist locally
        gzfile = self.get_path('mangagalaxyzoo', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not gzfile:
            gzfile = self.download_vac('mangagalaxyzoo', path_params=path_params)

        return gzdata


class GZData(object):
    ''' Row data from the summary file
    is returned via the `data` property.  

    '''

    def __init__(self, plateifu, allfile=None, specfile=None):
        self._gzfile = gzfile
        self._plateifu = plateifu
        self._gzdata = self._open_file(gzfile)
        self._indata = plateifu in self._gzdata['plateifu']
 
    @staticmethod
    def _open_file(gzfile):
        return astropy.io.fits.getdata(gzfile, 1)

    @property
    def data(self):
        ''' Returns the FITS row data from the Galaxy Zoo file '''

        if not self._indata:
            return "No Galaxy Zoo data exists for {0}".format(self._plateifu)

        idx = self._gzdata['plateifu'] == self._plateifu
        return self._gzdata[idx]



