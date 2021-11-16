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

    URL: https://www.sdss.org/dr17/data_access/value-added-catalogs/?vac_id=manga-morphologies-from-galaxy-zoo
    
    Description Returns Galaxy Zoo morphology for MaNGA galaxies. 
    The Galaxy Zoo (GZ) data for SDSS galaxies has been split over several iterations of www.galaxyzoo.org, 
    with the MaNGA target galaxies being spread over five different GZ data sets. In this value added catalog,
    for DR15, we bring all of these galaxies into one single catalog and re-run the debiasing code (Hart et al. 2016) in 
    a consistent manner across the all the galaxies. This catalog includes data from Galaxy Zoo 2 (previously 
    published in Willett et al. 2013) and newer data from Galaxy Zoo 4 (currently unpublished).

    For DR17, we provide new and updated Galaxy Zoo (GZ) data for the final MaNGA galaxies. This has been split over 
    three files, each corresponding to a separate GZ catalogue. We have `MaNGA_GZD_auto-v1_0_1.fits`, which
    corresponds to the automated classifications GZ DECaLS, described in Walmsley et al. 2021. There is also 
    `MaNGA_gzUKIDSS-v1_0_1.fits`, which correponds to GZ:UKIDSS. Finally, we have put the rest of GZ 
    (so not including GZ DECaLS and GZ:UKIDSS) in `MaNGA_gz-v2_0_1.fits`. For more information, please 
    refer to the datamodels provided.

    Authors: Coleman Krawczyk, Karen Masters, Tobias Géron and the rest of the Galaxy Zoo Team.

    """

    # Required parameters
    name = "galaxyzoo"
    description = "Returns Galaxy Zoo morphology"
    version = {"MPL-7": "v1_0_1", "MPL-8": "v1_0_1", "DR15": "v1_0_1", "DR16": "v1_0_1", 
               "MPL-11" : None, "DR17" : None}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube, marvin.tools.maps.Maps, marvin.tools.modelcube.ModelCube)

    # Required method
    def set_summary_file(self, release):
        ''' Sets the path to the GalaxyZoo summary file.
        
        Sets the paths to the GalaxyZoom summary file(s).  For DR15 this is a single summary file,
        while for DR17, this has been split into three files, so ``self.summary_file`` and 
        ``self.path_params`` return lists for DR17.
        '''

        self.summary_file = []
        self.path_params = []

        # define the variables to build a unique path to your VAC file
        if release in ["DR17", "MPL-11"]:  
            # for DR17, path is more complicated as we have three files.
            files = ['GZD_auto', 'gzUKIDSS', 'gz']
            version_DR17 = {"GZD_auto": "v1_0_1", "gzUKIDSS": "v1_0_1", "gz": "v2_0_1"}
            for file in files:
                params = {"file" : file, "ver" : version_DR17[file]}
                self.path_params.append(params)
                
                # get_path returns False if the files do not exist locally
                self.summary_file.append(self.get_path("mangagalaxyzoo", path_params=params))
        else: 
            # for other releases prior to DR17, simply do based on release as before
            self.path_params.append({"ver": self.version[release]})
            self.summary_file.append(self.get_path("mangagalaxyzoo", path_params=self.path_params[0]))

    # Required method
    def get_target(self, parent_object):
        ''' Accesses VAC data for a specific target from a Marvin Tool object '''

        # get any parameters you need from the parent object
        mangaid = parent_object.mangaid

        # download the vac from the SAS if it does not already exist locally
        for i in range(len(self.summary_file)):
            if not self.file_exists(self.summary_file[i]):
                self.summary_file[i] = self.download_vac("mangagalaxyzoo", path_params=self.path_params[i])
                
        # Open the file(s) using fits.getdata for extension 1
        data = []
        for i in range(len(self.summary_file)):
            data.append(astropy.io.fits.getdata(self.summary_file[i], 1))

        # Return selected line(s)
        result = {}
        keys = ['gz_auto','gz_ukidss', 'gz']

        for i in range(len(self.summary_file)):
            indata = mangaid in data[i]["mangaid"]
            if not indata:
                pass # passes instead of adding an empty key-value pair.

            else:
                idx = data[i]["mangaid"] == mangaid
                result[keys[i]] = data[i][idx]

        # return the result
        if len(self.summary_file) == 1:
            # to make sure output stays the same if version is < DR17
            return list(result.values())[0]
        else:
            # for DR17, return dict with classifications of all three files
            return result 


