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
import marvin.tools


from .base import VACMixIn


class VMORPHOVAC(VACMixIn):
    """Provides access to the MaNGA-VISUAL-MORPHOLOGY VAC.

    VAC name: manga_visual_morpho

    URL: https://data.sdss.org/datamodel/files/MANGA_MORPHOLOGY/visual_morpho/manga_Vmorpho.html

    Description: A new morphology catalogue is presented in this VAC, based on a pure visual morphological classification. This catalogue contains the T-Type morphology, visual attributes (barred, edge-on, tidal debris) and the CAS parameters (Concentration, Asymmetry and Clumpiness; from the DESI images.

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
        mosaic = self.get_path('mangaVmorphoImgs', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not allfile:
            allfile = self.download_vac('mangaVmorpho', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not mosaic:
            mosaic = self.download_vac('mangaVmorphoImgs', path_params=path_params)

        print("Mosaics are available",mosaic)

        # Returns the FITS row data from the manga_visual_morpho-1.0.1.fits file

        alldata = astropy.io.fits.getdata(allfile,1)
        idx = alldata['plateifu'] == plateifu
        return alldata[idx]





