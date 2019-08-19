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
from marvin.tools.quantities.spectrum import Spectrum
from marvin.utils.general.general import get_drpall_table
from marvin.utils.plot.scatter import plot as scatplot

from .base import VACMixIn


class FIREFLYVAC(VACMixIn):
    """Provides access to the MaNGA-FIREFLY VAC.

    VAC name: FIREFLY

    URL: https://www.sdss.org/dr15/manga/manga-data/manga-firefly-value-added-catalog/

    Description: Returns integrated and resolved stellar population parameters fitted by FIREFLY

    Authors: Jianhui Lian, Daniel Thomas, Claudia Maraston, and Lewis Hill

    """

    # Required parameters
    name = 'mangaffly'
    description = 'Returns stellar population parameters fitted by FIREFLY'
    version = {'DR15': 'v1_1_2'}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube,
               marvin.tools.maps.Maps,
               marvin.tools.modelcube.ModelCube)

    # Required method
    def get_data(self,parent_object):

        release = parent_object.release
        drpver = parent_object._drpver
        plateifu = parent_object.plateifu

        # define the variables to build a unique path to your VAC file
        path_params = {'ver': self.version[release],'drpver': self.drpver}
        # get_path returns False if the files do not exist locally
        allfile = self.get_path('mangaffly', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not allfile:
          allfile = self.download_vac('mangaffly', path_params=path_params)

        # create container for more complex return data. Parameters could be 'lw_age', 'mw_age', 'lw_z', 'mw_z'.
        ffly = FFLY(plateifu,allfile=allfile,parameter)

        return ffly


class FFLY(object):

    def __init__(self, plateifu, allfile=None,parameter=None):
        self._allfile = allfile
        self._plateifu = plateifu
        self._ffly_data = self._open_file(allfile)
        self._indata = plateifu in self._ffly_data[1].data['plateifu']
        self._parameter = parameter

    def _open_file(fflyfile):
        return astropy.io.fits.open(fflyfile)

    @property
    def global(self):
        ''' Returns the global stellar population properties from the mangafirefly summary file '''
        if not self._indata:
            return "No FIREFLY result exists for {0}".format(self._plateifu)

        idx = self._ffly_data[1].data['plateifu'] == self._plateifu

        return self._ffly_data[2].data[self._parameter+'_1re'][idx]

    def gradient(self):
        ''' Returns the gradient of stellar population properties from the mangafirefly summary file '''
        if not self._indata:
            return "No FIREFLY result exists for {0}".format(self._plateifu)

        idx = self._ffly_data[1].data['plateifu'] == self._plateifu
         
        return self._ffly_data[3].data[self._parameter+'_gradient'][idx]

    def plot_map(self):

        ''' plot map of stellar population properties'''
        if not self._indata:
            return "No FIREFLY result exists for {0}".format(self._plateifu)

        idx = self._ffly_data[1].data['plateifu'] == self._plateifu
        binid = self._ffly_data[5].data
        bin1d = self._ffly_data[4].data[idx,:,0][0]
        prop = self._ffly_data[self._parameter+'_voronoi'].data[idx,:,0][0]
        image_sz = 76

        maps = np.zeros((image_sz,image_sz))-99
        for i in range(image_sz):
         for j in range(image_sz):
          idbin = (bin1d==binid[idx,i,j])
          if len(bin1d[idbin])==1:
          maps[i,j] = prop[idbin]

        #only show the spaxels with non-empty values
        masked_array = np.ma.array(maps,mask=(maps<-10))

        #plot the map 
        f = plt.imshow(masked_array,interpolation='nearest',cmap='RdYlBu_r',origin='lower')

        return f
    


