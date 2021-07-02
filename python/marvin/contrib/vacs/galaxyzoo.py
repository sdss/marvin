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

    Description DR15: Returns Galaxy Zoo morphology for MaNGA galaxies. 
    The Galaxy Zoo (GZ) data for SDSS galaxies has been split over several iterations of www.galaxyzoo.org, 
    with the MaNGA target galaxies being spread over five different GZ data sets. In this value added catalog 
    we bring all of these galaxies into one single catalog and re-run the debiasing code (Hart et al. 2016) in 
    a consistent manner across the all the galaxies. This catalog includes data from Galaxy Zoo 2 (previously 
    published in Willett et al. 2013) and newer data from Galaxy Zoo 4 (currently unpublished).

    Description DR17: Returns Galaxy Zoo morphology for MaNGA galaxies.
    We provide new and updated Galaxy Zoo (GZ) data for the final MaNGA galaxies. This has been split over 
    three files, each corresponding to a separate GZ catalogue. We have `MaNGA_GZD_auto-v1_0_1.fits`, which
    corresponds to the automated classifications GZ DECaLS, described in Walmsley et al. 2021. There is also 
    `MaNGA_gzUKIDSS-v1_0_1.fits`, which correponds to GZ:UKIDSS. Finally, we have put the rest of GZ in 
    `MaNGA_gz-v2_0_1.fits`. For more information, please refer to the datamodels provided.


    Authors: Coleman Krawczyk, Karen Masters, Tobias Géron and the rest of the Galaxy Zoo Team.

    """

    # Required parameters
    name = "galaxyzoo"
    description = "Returns Galaxy Zoo morphology"
    version = {"MPL-7": "v1_0_1", "MPL-8": "v1_0_1", "DR15": "v1_0_1", "DR16": "v1_0_1"}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube, marvin.tools.maps.Maps, marvin.tools.modelcube.ModelCube)

    # Required method
    def set_summary_file(self, release,  file):
        ''' Sets the path to the GalaxyZoo summary file. File argument should either be 
        "GZD", "GZ_UKIDSS" or "GZ".'''

        # define the variables to build a unique path to your VAC file
        if release == "DR17": # path is more complicated in DR17, as we have three files.
            files = {"GZD" : "GZD_auto", "GZ_UKIDSS" : "gzUKIDSS", "GZ" : "gz"}
            version_DR17 = {"GZD" : "v1_0_1", "GZ_UKIDSS" : "v1_0_1", "GZ" : "v2_0_1"}
            self.path_params = {"file" : files[file], "ver" : version_DR17[file]}
            
        else: # if not DR17, simply do based on release
            self.path_params = {"ver": self.version[release]}

        # get_path returns False if the files do not exist locally
        self.summary_file = self.get_path("mangagalaxyzoo", path_params=self.path_params)

    # Required method
    def get_target(self, parent_object):
        ''' Accesses VAC data for a specific target from a Marvin Tool object '''

        # get any parameters you need from the parent object
        mangaid = parent_object.mangaid

        # download the vac from the SAS if it does not already exist locally
        if not self.file_exists(self.summary_file):
            self.summary_file = self.download_vac("mangagalaxyzoo", path_params=self.path_params)

        # Open the file using fits.getdata for extension 1
        data = astropy.io.fits.getdata(self.summary_file, 1)

        # Return selected line(s)
        indata = mangaid in data["mangaid"]
        if not indata:
            return "No Galaxy Zoo data exists for {0}".format(mangaid)

        idx = data["mangaid"] == mangaid
        return data[idx]

