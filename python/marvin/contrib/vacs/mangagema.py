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

from astropy.io import fits
from astropy.table import Table
import numpy as np
import marvin.tools


from .base import VACMixIn


class GEMAVAC(VACMixIn):
    """Provides access to the MaNGA-GEMA VAC.

    VAC name: GEMA

    URL: https://www.sdss.org/dr15/data_access/value-added-catalogs/?vac_id=gema-vac-galaxy-environment-for-manga-value-added-catalog

    Description: The GEMA VAC contains many different quantifications of the local and the large-scale environments for MaNGA galaxies. Please visit the DATAMODEL at https://data.sdss.org/datamodel/files/MANGA_GEMA/GEMA_VER to see the description of each table composing the catalogue.

    Authors: Maria Argudo-Fernandez, Daniel Goddard, Daniel Thomas, Zheng Zheng, Lihwai Lin, Ting Xiao, Fangting Yuan, Jianhui Lian, et al

    """

    # Required parameters
    name = 'mangagema'
    description = 'Returns GEMA table data'
    version = {'MPL-7': '1.0.1',
               'DR15': '1.0.1'}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube,
               marvin.tools.maps.Maps,
               marvin.tools.modelcube.ModelCube)

    # Required method
    def get_data(self, parent_object):
        
        # get any parameters you need from the parent object
        mangaid = parent_object.mangaid
        release = parent_object.release
        
        # define the variables to build a unique path to your VAC file
        path_params = {'ver': self.version[release]}

        # get_path returns False if the files do not exist locally
        gemafile = self.get_path('mangagema', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not gemafile:
            gemafile = self.download_vac('mangagema', path_params=path_params)
        
        # opening tables in VAC file
        completeness = fits.getdata(gemafile, 1)
        LSS_1_all = fits.getdata(gemafile, 2)
        LSS_1_002 = fits.getdata(gemafile, 3)
        LSS_1_006 = fits.getdata(gemafile, 4)
        LSS_1_010 = fits.getdata(gemafile, 5)
        LSS_1_015 = fits.getdata(gemafile, 6)
        LSS_5_all = fits.getdata(gemafile, 7)
        LSS_5_002 = fits.getdata(gemafile, 8)
        LSS_5_006 = fits.getdata(gemafile, 9)
        LSS_5_010 = fits.getdata(gemafile, 10)
        LSS_5_015 = fits.getdata(gemafile, 11)
        pairs = fits.getdata(gemafile, 12)
        groups = fits.getdata(gemafile, 13)
        overdensity = fits.getdata(gemafile, 14)
        LSS_tensor = fits.getdata(gemafile, 15)
        
        # Return selected line(s)
        indata = mangaid in completeness['mangaid']
        if not indata:
            return "No LSS data exists for {0}".format(mangaid)
        
        indata = mangaid in pairs['mangaid']
        if not indata:
            return "No pair data exists for {0}".format(mangaid)
        
        indata = mangaid in groups['mangaid']
        if not indata:
            return "No group data exists for {0}".format(mangaid)
        
        indata = mangaid in overdensity['mangaid']
        if not indata:
            return "No overdensity data exists for {0}".format(mangaid)
        
        indata = mangaid in LSS_tensor['mangaid']
        if not indata:
            return "No structure data exists for {0}".format(mangaid)
        
        # creating an index to select the data for a mangaid galaxy
        indexLSS = completeness['mangaid'] == mangaid
        indexpairs = pairs['mangaid'] == mangaid
        indexgroups = groups['mangaid'] == mangaid
        indexover = overdensity['mangaid'] == mangaid
        indextensor = LSS_tensor['mangaid'] == mangaid
        
        gema_data = (completeness[indexLSS], LSS_1_all[indexLSS], LSS_1_002[indexLSS], LSS_1_006[indexLSS], LSS_1_010[indexLSS], LSS_1_015[indexLSS], LSS_5_all[indexLSS], LSS_5_002[indexLSS], LSS_5_006[indexLSS], LSS_5_010[indexLSS], LSS_5_015[indexLSS], pairs[indexpairs], groups[indexgroups], overdensity[indexover], LSS_tensor[indextensor])
        
        return gema_data
