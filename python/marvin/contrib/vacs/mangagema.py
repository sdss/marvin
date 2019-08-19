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


    def get_gema_data(allfile, mangaid):
        
        # opening VAC file
        gemafile = fits.open(allfile)
        
        # creating an index to select the data for a mangaid galaxy
        indexLSS = np.where(gemafile[1].data['mangaid'] == mangaid)
        indexpairs = np.where(gemafile[12].data['mangaid'] == mangaid)
        indexgroups = np.where(gemafile[13].data['mangaid'] == mangaid)
        indexover = np.where(gemafile[14].data['mangaid'] == mangaid)
        indextensor = np.where(gemafile[15].data['mangaid'] == mangaid)
        
        
        # Transforming data each BinTableHDU in the VAC file to an astropy table for the mangaid galaxy
        completeness = Table(gemafile[1].data[indexLSS])
        LSS_1_all = Table(gemafile[2].data[indexLSS])
        LSS_1_002 = Table(gemafile[3].data[indexLSS])
        LSS_1_006 = Table(gemafile[4].data[indexLSS])
        LSS_1_010 = Table(gemafile[5].data[indexLSS])
        LSS_1_015 = Table(gemafile[6].data[indexLSS])
        LSS_5_all = Table(gemafile[7].data[indexLSS])
        LSS_5_002 = Table(gemafile[8].data[indexLSS])
        LSS_5_006 = Table(gemafile[9].data[indexLSS])
        LSS_5_010 = Table(gemafile[10].data[indexLSS])
        LSS_5_015 = Table(gemafile[11].data[indexLSS])
        pairs = Table(gemafile[12].data[indexpairs])
        groups = Table(gemafile[13].data[indexgroups])
        overdensity = Table(gemafile[14].data[indexover])
        LSS_tensor = Table(gemafile[15].data[indextensor])
        
        # closing VAC file
        gemafile.close()
        
        gemadata = (completeness, LSS_1_all, LSS_1_002, LSS_1_006, LSS_1_010, LSS_1_015, LSS_5_all, LSS_5_002, LSS_5_006, LSS_5_010, LSS_5_015, pairs, groups, overdensity, LSS_tensor)
        
        return gemadata



    # Required method
    def get_data(self, parent_object):
        
        # get any parameters you need from the parent object
        mangaid = parent_object.mangaid
        release = parent_object.release
        
        # define the variables to build a unique path to your VAC file
        path_params = {'ver': self.version[release]}

        # get_path returns False if the files do not exist locally
        allfile = self.get_path('mangagema', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not allfile:
            allfile = self.download_vac('mangagema', path_params=path_params)
            
        gema_data = get_gema_data(allfile, mangaid)    

        return gema_data









