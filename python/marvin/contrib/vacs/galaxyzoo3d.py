#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-03
# @Filename: galaxyzoo3d.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego
# @Last modified time: 2018-07-09 10:31:53


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

    name = 'galaxyzoo3d'
    version = 'v1_0_0'

    include = (marvin.tools.cube.Cube,
               marvin.tools.maps.Maps,
               marvin.tools.modelcube.ModelCube)

    def get_data(self, parent_object):

        mangaid = parent_object.mangaid.strip()
        self.set_sandbox_path('galaxyzoo3d/{0}/{1}_*.fits.gz'.format(self.version, mangaid))

        if not self.file_exists():
            filename = self.download_vac()
        else:
            filename = self.get_path()

        galaxy_data = astropy.io.fits.open(filename)

        return galaxy_data
