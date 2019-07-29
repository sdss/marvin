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

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from .base import VACMixIn


class VMORPHOVAC(VACMixIn):
    """Provides access to the MaNGA-VISUAL-MORPHOLOGY VAC.

    VAC name: manga_visual_morpho

    URL: link

    Description: Returns visual morphology data

    Authors: J. Antonio Vazquez-Mata and Hector Hernandez-Toledo

    """

    # Required parameters
    name = 'mangaVM'
    description = 'Returns visual morphology data'
    version = {'MPL-7': '1.0.1',
               'DR15': '1.0.1'}

    ## optional Marvin Tools to attach your vac to
    #    include = (marvin.tools.cube.Cube,
    #           marvin.tools.maps.Maps,
    #           marvin.tools.modelcube.ModelCube)

    # Required method
    def get_data(self, parent_object):

        # get any parameters you need from the parent object
        plateifu = parent_object.plateifu
        release = parent_object.release

        # define the variables to build a unique path to your VAC file
        path_params = {'vmver': self.version[release],
                       'plateifu': plateifu, 'survey': sdss}

        # get_path returns False if the files do not exist locally
        allfile = self.get_path('mangaVmorpho', path_params=path_params)
        mosaicfile = self.get_path('mangaVmorphoImgs', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not allfile:
            allfile = self.download_vac('mangaVmorpho', path_params=path_params)

        # create container for more complex return data
        hidata = HIData(plateifu, allfile=allfile, mosaicfile=mosaicfile)

        # get the spectral data for that row if it exists
        if hidata._indata and not mosaicfile:
            hidata._mosaicfile = self.download_vac('mangaVmorphoImgs', path_params=path_params)

        return hidata


class HIData(object):
    ''' A customized class to handle more complex data

    This class handles data from both the HI summary file and the
    individual HI spectral files.  Row data from the summary file
    is returned via the `data` property.  Spectral data can be plotted via
    the `plot_spectrum` method.

    '''

    def __init__(self, plateifu, allfile=None, mosaicfile=None):
        self._allfile = allfile
        self._mosaicfile = mosaicfile
        self._plateifu = plateifu
        self._hi_data = self._open_file(allfile)
        self._indata = plateifu in self._hi_data['plateifu']
    #        self._specdata = None

    def __repr__(self):
        return 'HI({0})'.format(self._plateifu)

#    @staticmethod
#    def _open_file(hifile):
#        return astropy.io.fits.getdata(hifile, 1)

#    @property
#    def data(self):
#        ''' Returns the FITS row data from the mangaHIall summary file '''
#
#        if not self._indata:
#            return "No HI data exists for {0}".format(self._plateifu)
#
#        idx = self._hi_data['plateifu'] == self._plateifu
#        return self._hi_data[idx]

    def plot_mosaic(self):
        ''' Plot the mosaic '''

        if self._mosaicfile:

            img = mpimg.imread(self._mosaicfile)
            ax = plt.imshow(img)
            return ax
        return None
