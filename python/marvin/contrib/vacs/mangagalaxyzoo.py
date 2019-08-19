# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Modified from mangahi.py by Karen Masters. 

from __future__ import print_function, division, absolute_import

import astropy
import marvin.tools

from .base import VACMixIn


class GZVAC(VACMixIn):
    """Provides access to the MaNGA Galaxy Zoo Morphology VAC.

    VAC name: MaNGA Morphologies from Galaxy Zoo

    URL: https://www.sdss.org/dr15/data_access/value-added-catalogs/?vac_id=manga-morphologies-from-galaxy-zoo

    Description: Returns Galaxy Zoo morphology for MaNGA galaxies. 
    The Galaxy Zoo (GZ) data for SDSS galaxies has been split over several iterations of www.galaxyzoo.org, 
    with the MaNGA target galaxies being spread over five different GZ data sets. In this value added catalog 
    we bring all of these galaxies into one single catalog and re-run the debiasing code (Hart et al. 2016) in 
    a consistent manner across the all the galaxies. This catalog includes data from Galaxy Zoo 2 (previously 
    published in Willett et al. 2013) and newer data from Galaxy Zoo 4 (currently unpublished).

    Authors: Coleman Krawczyk, Karen Masters and the rest of the Galaxy Zoo Team.

    """

    # Required parameters
    name = 'galaxyzoo'
    description = 'Returns Galaxy Zoo morphology'
    version = {'MPL-7': 'v1_0_1', 'MPL-8': 'v1_0_1',
               'DR15': 'v1_0_1', 'DR16': 'v1_0_1'}

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
        gzfile = self.get_path('mangagalaxyzoo', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not gzfile:
            gzfile = self.download_vac('mangagalaxyzoo', path_params=path_params)
            
        # Open the file    
        data = astropy.io.fits.getdata(gzfile, 1)
        
        # Return selected line(s)
        indata = mangaid in data['mangaid']
        if not indata:
            return "No Galaxy Zoo data exists for {0}".format(mangaid)

        idx = data['mangaid'] == mangaid
        return data[idx]


# class GZData(object):
#     ''' Row data from the summary file
#     is returned via the `data` property.  

#     '''

#     def __init__(self, plateifu, allfile=None, specfile=None):
#         self._gzfile = gzfile
#         self._plateifu = plateifu
#         self._gzdata = self._open_file(gzfile)
#         self._indata = plateifu in self._gzdata['plateifu']
 
#     @staticmethod
#     def _open_file(gzfile):
#         return astropy.io.fits.getdata(gzfile, 1)

#     @property
#     def data(self):
#         ''' Returns the FITS row data from the Galaxy Zoo file '''

#         if not self._indata:
#             return "No Galaxy Zoo data exists for {0}".format(self._plateifu)

#         idx = self._gzdata['plateifu'] == self._plateifu
#         return self._gzdata[idx]



