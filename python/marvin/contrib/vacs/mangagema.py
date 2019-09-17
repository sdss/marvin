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
import marvin.tools
from marvin import log

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
    version = {'MPL-7': '1.0.1', 'DR15': '1.0.1'}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube, marvin.tools.maps.Maps, marvin.tools.modelcube.ModelCube)

    # Required method
    def get_data(self, parent_object):

        # Create a dictionary for GEMA VAC data
        gemadata = {
            "completeness": "",
            "LSS_1_all": "",
            "LSS_1_002": "",
            "LSS_1_006": "",
            "LSS_1_010": "",
            "LSS_1_015": "",
            "LSS_5_all": "",
            "LSS_5_002": "",
            "LSS_5_006": "",
            "LSS_5_010": "",
            "LSS_5_015": "",
            "pairs": "",
            "groups": "",
            "overdensity": "",
            "LSS_tensor": "",
        }

        # get any parameters you need from the parent object
        mangaid = parent_object.mangaid
        release = parent_object.release

        # define the variables to build a unique path to your VAC file
        path_params = {'ver': self.version[release]}

        # get_path returns False if the files do not exist locally
        gemapath = self.get_path('mangagema', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not gemapath:
            gemapath = self.download_vac('mangagema', path_params=path_params)

        # opening tables in VAC file
        gemafile = fits.open(gemapath)

        # Return selected line(s) for a mangaid galaxy
        # LSS parameters
        indata = mangaid in gemafile[1].data['mangaid']
        if not indata:
            log.warning("No LSS data exists for {0}".format(mangaid))
        else:
            log.info("LSS data exists for {0}".format(mangaid))
            log.warning(
                "Warning: Do not use LSS parameters defined in volume with z < z_manga_galaxy"
            )
            indexLSS = gemafile[1].data['mangaid'] == mangaid
            gemadata["completeness"] = gemafile[1].data[indexLSS]
            gemadata["LSS_1_all"] = gemafile[2].data[indexLSS]
            gemadata["LSS_1_002"] = gemafile[3].data[indexLSS]
            gemadata["LSS_1_006"] = gemafile[4].data[indexLSS]
            gemadata["LSS_1_010"] = gemafile[5].data[indexLSS]
            gemadata["LSS_1_015"] = gemafile[6].data[indexLSS]
            gemadata["LSS_5_all"] = gemafile[7].data[indexLSS]
            gemadata["LSS_5_002"] = gemafile[8].data[indexLSS]
            gemadata["LSS_5_006"] = gemafile[9].data[indexLSS]
            gemadata["LSS_5_010"] = gemafile[10].data[indexLSS]
            gemadata["LSS_5_015"] = gemafile[11].data[indexLSS]

        # Local environment: close pair galaxies
        indata = mangaid in gemafile[12].data['mangaid']
        if not indata:
            log.warning("No pair data exists for {0}".format(mangaid))
        else:
            log.info("Pair data exists for {0}".format(mangaid))
            indexpair = gemafile[12].data['mangaid'] == mangaid
            gemadata["pairs"] = gemafile[12].data[indexpair]

        # Local environment: group galaxies
        indata = mangaid in gemafile[13].data['mangaid']
        if not indata:
            log.warning("No group data exists for {0}".format(mangaid))
        else:
            log.info("Group data exists for {0}".format(mangaid))
            indexgroup = gemafile[13].data['mangaid'] == mangaid
            gemadata["groups"] = gemafile[13].data[indexgroup]

        # LSS overdensity-corrected local density
        indata = mangaid in gemafile[14].data['mangaid']
        if not indata:
            log.warning("No overdensity data exists for {0}".format(mangaid))
        else:
            log.info("Overdensity data exists for {0}".format(mangaid))
            indexover = gemafile[14].data['mangaid'] == mangaid
            gemadata["overdensity"] = gemafile[14].data[indexover]

        # LSS structure
        indata = mangaid in gemafile[15].data['mangaid']
        if not indata:
            log.warning("No structure data exists for {0}".format(mangaid))
        else:
            log.info("Structure data exists for {0}".format(mangaid))
            indextensor = gemafile[15].data['mangaid'] == mangaid
            gemadata["LSS_tensor"] = gemafile[15].data[indextensor]

        gema_data = gemadata

        # closing the FITS file
        gemafile.close()

        return gema_data
