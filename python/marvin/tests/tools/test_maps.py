#!/usr/bin/env python
# encoding: utf-8
#
# test_maps.py
#
# Created by José Sánchez-Gallego on 22 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import pytest


import astropy.io.fits
import numpy as np

import marvin
import marvin.tests
import marvin.tools.map
import marvin.tools.maps
import marvin.tools.spaxel

from marvin.tests import marvin_skip_if, marvin_skip_if_class


def _assert_maps(maps, galaxy):
    """Basic checks for a Maps object."""

    assert maps is not None
    assert maps.plateifu == galaxy.plateifu
    assert maps.mangaid == galaxy.mangaid
    assert maps.wcs is not None
    assert maps.bintype == galaxy.bintype

    assert len(maps.shape) == len(galaxy.shape)
    for ii in range(len(maps.shape)):
        assert maps.shape[ii] == galaxy.shape[ii]


@marvin_skip_if_class(galaxy=dict(template=['MILES-THIN', 'MIUSCAT-THIN']))
@marvin_skip_if_class(galaxy=dict(bintype='NONE', mode='include'))
class TestMaps(object):

    def test_load(self, galaxy, data_origin):

        if data_origin == 'file':
            maps_kwargs = dict(filename=galaxy.mapspath)
        else:
            maps_kwargs = dict(plateifu=galaxy.plateifu, release=galaxy.release,
                               bintype=galaxy.bintype, template_kin=galaxy.template,
                               mode='local' if data_origin == 'db' else 'remote')

        maps = marvin.tools.maps.Maps(**maps_kwargs)

        _assert_maps(maps, galaxy)

        assert maps.data_origin == data_origin

        if data_origin == 'file':
            assert isinstance(maps.data, astropy.io.fits.HDUList)
        elif data_origin == 'db':
            assert isinstance(maps.data, marvin.marvindb.dapdb.File)

        assert maps.cube is not None
        assert maps.cube.plateifu == galaxy.plateifu
        assert maps.cube.mangaid == galaxy.mangaid

    # @marvin_test_if(mode='include', releases=['MPL-4'])
    # def test_load_file_mpl4_global_mpl5(self, galaxy):
    #
    #     marvin.config.setMPL('MPL-5')
    #     maps = marvin.tools.maps.Maps(filename=galaxy.mapspath)
    #     assert maps._release == 'MPL-4'
    #     assert maps._drpver == 'v1_5_1'
    #     assert maps._dapver == '1.1.1'
    #
    # # TODO: generalise this test to some (if not all) releases and bintypes
    # @marvin_test_if(mode='include', releases=['MPL-4'], bintypes=['NONE'])
    # def test_get_spaxel_file(self, galaxy):
    #
    #     maps = marvin.tools.maps.Maps(filename=galaxy.mapspath)
    #     spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower')
    #
    #     assert isinstance(spaxel, marvin.tools.spaxel.Spaxel)
    #     assert spaxel.spectrum is not None
    #     assert len(spaxel.properties.keys()) > 0
    #     pytest.approx(spaxel.properties['stellar_vel'].ivar, 1.013657e-05)
    #
    # def test_get_spaxel_test2_file(self, galaxy):
    #
    #     maps = marvin.tools.maps.Maps(filename=galaxy.mapspath)
    #     spaxel = maps.getSpaxel(x=5, y=5)
    #
    #     assert isinstance(spaxel, marvin.tools.spaxel.Spaxel)
    #     assert spaxel.spectrum is not None
    #     assert len(spaxel.properties.keys()) > 0
    #
    # def test_get_spaxel_no_db(self, galaxy):
    #     """Tests getting an spaxel if there is no DB."""
    #
    #     marvin.marvindb.db = None
    #
    #     maps = marvin.tools.maps.Maps(filename=galaxy.mapspath)
    #     spaxel = maps.getSpaxel(x=5, y=5)
    #
    #     assert isinstance(spaxel, marvin.tools.spaxel.Spaxel)
    #     assert spaxel.spectrum is not None
    #     assert len(spaxel.properties.keys()) > 0


# class TestMapsDB(object):
#
#     def test_load_from_db_with_release(self, galaxy):
#
#         maps = marvin.tools.maps.Maps(plateifu=galaxy.plateifu, release=galaxy.release,
#                                       bintype=galaxy.bintype, template_kin=galaxy.template,
#                                       mode='local')
#         _assert_maps(maps, galaxy)
#         assert maps.data is not None
#         assert isinstance(maps.data, marvin.marvindb.dapdb.File)
#         assert maps.bintype.lower() == galaxy.bintype.lower()
#         assert maps.template_kin.lower() == galaxy.template.lower()
#
#     def test_load_from_db_with_config(self, galaxy):
#
#         marvin.config.release = galaxy.release
#         maps = marvin.tools.maps.Maps(plateifu=galaxy.plateifu,
#                                       bintype=galaxy.bintype, template_kin=galaxy.template,
#                                       mode='local')
#         _assert_maps(maps, galaxy)
#         assert maps.data is not None
#         assert isinstance(maps.data, marvin.marvindb.dapdb.File)
#         assert maps.bintype.lower() == galaxy.bintype.lower()
#         assert maps.template_kin.lower() == galaxy.template.lower()

#     def test_get_spaxel_db(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='local', release='MPL-4')
#         spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower', spectrum=False)
#
#         self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
#         self.assertIsNone(spaxel.spectrum)
#         self.assertTrue(len(spaxel.properties.keys()) > 0)
#
#         self.assertAlmostEqual(spaxel.properties['stellar_vel'].ivar, 1.013657e-05)
#
#     def test_get_spaxel_test2_db(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='local')
#         spaxel = maps.getSpaxel(x=5, y=5)
#
#         self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
#         self.assertIsNotNone(spaxel.spectrum)
#         self.assertTrue(len(spaxel.properties.keys()) > 0)
#
#     def test_get_spaxel_getitem(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='local')
#         spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower')
#         spaxel_getitem = maps[8, 15]
#
#         self.assertTrue(isinstance(spaxel_getitem, marvin.tools.spaxel.Spaxel))
#         self.assertIsNotNone(spaxel_getitem.spectrum)
#         self.assertTrue(len(spaxel_getitem.properties.keys()) > 0)
#
#         self.assertAlmostEqual(spaxel_getitem.spectrum.flux[100], spaxel.spectrum.flux[100])
#
#     def test_bintype_maps_db_vor10(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, bintype='VOR10', release='MPL-5')
#         self.assertEqual(maps.bintype, 'VOR10')
#
#     def test_bintype_maps_db_spx(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, release='MPL-5')
#         self.assertEqual(maps.bintype, 'SPX')
#
#
# class TestMapsAPI(object):
#
#     def test_load_default_from_api(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote')
#         self._assert_maps(maps)
#         self.assertIsNone(maps.data)
#         self.assertIsNotNone(maps.cube)
#         self.assertEqual(maps.cube.data_origin, 'db')
#         self.assertEqual(maps.cube.plateifu, self.plateifu)
#         self.assertEqual(maps.cube.mangaid, self.mangaid)
#
#     def test_load_full_from_api(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu,
#                                       bintype='none', template_kin='MILES-THIN', mode='remote')
#         self._assert_maps(maps)
#         self.assertIsNone(maps.data)
#         self.assertEqual(maps.data_origin, 'api')
#         self.assertTrue(maps.bintype, 'NONE')
#         self.assertTrue(maps.template_kin, 'MILES-THIN')
#
#     def test_get_spaxel_api(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote', release='MPL-4')
#         spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower')
#
#         self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
#         self.assertIsNotNone(spaxel.spectrum)
#         self.assertTrue(len(spaxel.properties.keys()) > 0)
#
#         self.assertAlmostEqual(spaxel.properties['stellar_vel'].ivar, 1.013657e-05)
#
#     def test_get_spaxel_test2_api(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote')
#         spaxel = maps.getSpaxel(x=5, y=5)
#
#         self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
#         self.assertIsNotNone(spaxel.spectrum)
#         self.assertTrue(len(spaxel.properties.keys()) > 0)
#
#     def test_get_spaxel_drp_differ_from_global_api(self):
#
#         #self._update_release('MPL-5')
#         marvin.config.setMPL('MPL-5')
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote', release='MPL-4')
#         spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower', spectrum=False)
#
#         self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
#         self.assertIsNone(spaxel.spectrum)
#         self.assertTrue(len(spaxel.properties.keys()) > 0)
#
#         self.assertAlmostEqual(spaxel.properties['stellar_vel'].ivar, 1.013657e-05)
#
#     def test_bintype_maps_remote_vor10(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, bintype='VOR10',
#                                       release='MPL-5', mode='remote')
#         self.assertEqual(maps.bintype, 'VOR10')
#
#     def test_bintype_maps_remote_spx(self):
#
#         maps = marvin.tools.maps.Maps(plateifu=self.plateifu, release='MPL-5', mode='remote')
#         self.assertEqual(maps.bintype, 'SPX')
