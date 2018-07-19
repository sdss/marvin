#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-03
# @Filename: galaxyzoo3d.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by:   Brian Cherinka
# @Last modified time: 2018-07-09 17:23:24


from __future__ import absolute_import, division, print_function

import astropy

import marvin.tools

from .base import VACMixIn


class GalaxyZoo3DVAC(VACMixIn):
    """Provides access to the Galaxy Zoo: 3D VAC.

    VAC name: galaxyzoo3d

    URL: https://www.zooniverse.org/projects/klmasters/galaxy-zoo-3d

    Description: Returns people-powered classifications of internal structures
    of MaNGA galaxies.

    """

    # Required parameters
    name = 'galaxyzoo3d'
    description = 'Returns people-powered classifications of internal structures of MaNGA galaxies.'
    version = {'MPL-6': 'v1_0_0',
               'DR15': 'v1_0_0'}

    # optional Marvin Tools to attach your vac to
    include = (marvin.tools.cube.Cube,
               marvin.tools.maps.Maps,
               marvin.tools.modelcube.ModelCube)

    # Required method
    def get_data(self, parent_object):

        # get any parameters you need from the parent object
        mangaid = parent_object.mangaid.strip()
        release = parent_object.release

        # define the variables to build a unique path to your VAC file
        path_params = {'gz3dver': self.version[release], 'mangaid': mangaid,
                       'ifusize': '*', 'zooid': '*'}

        # get_path returns False if the files do not exist locally
        filename = self.get_path('mangagalaxyzoo3d', path_params=path_params)

        # download the vac from the SAS if it does not already exist locally
        if not filename:
            filename = self.download_vac('mangagalaxyzoo3d', path_params=path_params)

        # define your custom return data
        galaxy_data = astropy.io.fits.open(filename)

        return galaxy_data
