#!/usr/bin/env python
# encoding: utf-8
#
# test_spaxels.py
#
# Created by José Sánchez-Gallego on Sep 9, 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import unittest

import marvin
import marvin.tests
import marvin.tools.cube
import marvin.tools.maps
import marvin.tools.spaxel


class TestSpaxelBase(marvin.tests.MarvinTest):
    """Defines the files and plateifus we will use in the tests."""

    @classmethod
    def setUpClass(cls):

        cls.drpver_out = 'v1_5_1'
        cls.dapver_out = '1.1.1'

        cls.plate = 8485
        cls.mangaid = '1-209232'
        cls.plateifu = '8485-1901'

        cls.filename_cube = os.path.join(
            os.getenv('MANGA_SPECTRO_REDUX'), cls.drpver_out,
            '8485/stack/manga-8485-1901-LOGCUBE.fits.gz')

        cls.filename_maps_default = os.path.join(
            os.getenv('MANGA_SPECTRO_ANALYSIS'), cls.drpver_out, cls.dapver_out,
            'default', str(cls.plate), 'mangadap-{0}-default.fits.gz'.format(cls.plateifu))

        cls.marvindb_session = marvin.marvindb.session

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):

        marvin.marvindb.session = self.marvindb_session
        marvin.config.setMPL('MPL-4')

        self.assertTrue(os.path.exists(self.filename_cube))
        self.assertTrue(os.path.exists(self.filename_maps_default))

    def tearDown(self):
        pass


class TestSpaxelParent(TestSpaxelBase):

    def test_parent_shape_file(self):
        spaxel = marvin.tools.spaxel.Spaxel(filename=self.filename_cube, x=5, y=3)
        self.assertIsNotNone(spaxel._parent_shape)
        self.assertListEqual(list(spaxel._parent_shape), [34, 34])

    def test_parent_shape_db(self):
        spaxel = marvin.tools.spaxel.Spaxel(plateifu=self.plateifu, x=5, y=3)
        self.assertIsNotNone(spaxel._parent_shape)
        self.assertListEqual(list(spaxel._parent_shape), [34, 34])

    def test_parent_shape_remote(self):
        spaxel = marvin.tools.spaxel.Spaxel(plateifu=self.plateifu, x=5, y=3, mode='remote')
        self.assertIsNotNone(spaxel._parent_shape)
        self.assertListEqual(list(spaxel._parent_shape), [34, 34])

    def _test_from_cube_or_map(self, obj):
        spaxels = obj.getSpaxel(x=[5, 10], y=[2, 3])
        for sp in spaxels:
            self.assertIsNotNone(sp._parent_shape)
            self.assertListEqual(list(sp._parent_shape), [34, 34])

    def test_parent_spaxel_from_cube_file(self):
        cube = marvin.tools.cube.Cube(filename=self.filename_cube)
        self._test_from_cube_or_map(cube)

    def test_parent_spaxel_from_cube_db(self):
        cube = marvin.tools.cube.Cube(plateifu=self.plateifu)
        self.assertEqual(cube.data_origin, 'db')
        self._test_from_cube_or_map(cube)

    def test_parent_spaxel_from_cube_remote(self):
        cube = marvin.tools.cube.Cube(plateifu=self.plateifu, mode='remote')
        self.assertEqual(cube.data_origin, 'api')
        self._test_from_cube_or_map(cube)

    def test_parent_spaxel_from_maps_file(self):
        maps = marvin.tools.maps.Maps(filename=self.filename_maps_default)
        self._test_from_cube_or_map(maps)

    def test_parent_spaxel_from_maps_db(self):
        maps = marvin.tools.maps.Maps(plateifu=self.plateifu)
        self.assertEqual(maps.data_origin, 'db')
        self._test_from_cube_or_map(maps)

    def test_parent_spaxel_from_maps_remote(self):
        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote')
        self.assertEqual(maps.data_origin, 'api')
        self._test_from_cube_or_map(maps)

    def test_repr_central_spaxel(self):
        cube = marvin.tools.cube.Cube(plateifu=self.plateifu)
        spaxel = cube.getSpaxel(x=0, y=0, xyorig='center')
        self.assertEqual(repr(spaxel), '<Marvin Spaxel (x=17, y=17; x_cen=0, y_cen=0>')

    def test_repr_random_spaxel(self):
        cube = marvin.tools.cube.Cube(plateifu=self.plateifu)
        spaxel = cube.getSpaxel(x=5, y=3, xyorig='lower')
        self.assertEqual(repr(spaxel), '<Marvin Spaxel (x=5, y=3; x_cen=-12, y_cen=-14>')


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
