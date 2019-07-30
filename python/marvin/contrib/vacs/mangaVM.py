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

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube,
               marvin.tools.maps.Maps)
    #           marvin.tools.modelcube.ModelCube)

    # Required method
    def get_data(self, parent_object):

        # get any parameters you need from the parent object
        plateifu = parent_object.plateifu
        release = parent_object.release

        # define the variables to build a unique path to your VAC file
        path_params = {'vmver': self.version[release],
                       'plateifu': plateifu, 'survey': '*'}

        # get_path returns False if the files do not exist locally
        allfile = self.get_path('mangaVmorpho', path_params=path_params)
        sdss_img = self.get_path('mangaVmorphoImgs', path_params=path_params.update({'survey':'sdss'}))
        desi_img = self.get_path('mangaVmorphoImgs', path_params=path_params.update({'survey':'desi'}))

        # download the vac from the SAS if it does not already exist locally
        if not allfile:
            allfile = self.download_vac('mangaVmorpho', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not sdss_img or not desi_img:
            sdss_img = self.download_vac('mangaVmorpho', path_params=path_params.update({'survey':'sdss'}))
            desi_img = self.download_vac('mangaVmorpho', path_params=path_params.update({'survey':'desi'}))
                                                                                
        alldata = astropy.io.fits.getdata(allfile,1)
                                                                                        
        def data(self):
            ''' Returns the FITS row data from the manga_visual_morpho-1.0.1.fits file '''
            idx = alldata['plateifu'] == plateifu
            return alldata[idx]


        def show_mosaic(self):
            ''' Show the mosaic '''

            img = mpimg.imread(desi_img)
            plt.imshow(img)




