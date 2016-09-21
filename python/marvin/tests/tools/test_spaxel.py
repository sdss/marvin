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

from marvin.tools.analysis_props import DictOfProperties
from marvin.tools.spaxel import Spaxel
from marvin.tools.spectrum import Spectrum


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


class TestSpaxelInit(TestSpaxelBase):

    def test_no_cube_no_maps_db(self):

        spaxel = Spaxel(x=15, y=16, plateifu=self.plateifu)

        self.assertIsInstance(spaxel.cube, marvin.tools.cube.Cube)
        self.assertIsInstance(spaxel.maps, marvin.tools.maps.Maps)

        self.assertIsInstance(spaxel.spectrum, Spectrum)
        self.assertTrue(len(spaxel.properties) > 0)

    def test_cube_false_no_maps_db(self):

        spaxel = Spaxel(x=15, y=16, plateifu=self.plateifu, cube=False)

        self.assertIsNone(spaxel.cube)
        self.assertIsInstance(spaxel.maps, marvin.tools.maps.Maps)

        self.assertIsNone(spaxel.spectrum)
        self.assertTrue(len(spaxel.properties) > 0)

    def test_no_cube_maps_false_db(self):

        spaxel = Spaxel(x=15, y=16, plateifu=self.plateifu, maps=False)

        self.assertIsNone(spaxel.maps)
        self.assertIsInstance(spaxel.cube, marvin.tools.cube.Cube)

        self.assertIsInstance(spaxel.spectrum, Spectrum)
        self.assertTrue(len(spaxel.properties) == 0)

    def test_cube_object_db(self):

        cube = marvin.tools.cube.Cube(plateifu=self.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube)

        self.assertIsInstance(spaxel.cube, marvin.tools.cube.Cube)
        self.assertIsInstance(spaxel.maps, marvin.tools.maps.Maps)

        self.assertIsInstance(spaxel.spectrum, Spectrum)
        self.assertTrue(len(spaxel.properties) > 0)

    def test_cube_object_maps_false_db(self):

        cube = marvin.tools.cube.Cube(plateifu=self.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=False)

        self.assertIsInstance(spaxel.cube, marvin.tools.cube.Cube)
        self.assertIsNone(spaxel.maps)

        self.assertIsInstance(spaxel.spectrum, Spectrum)
        self.assertTrue(len(spaxel.properties) == 0)

    def test_cube_maps_object_filename(self):

        cube = marvin.tools.cube.Cube(filename=self.filename_cube)
        maps = marvin.tools.maps.Maps(filename=self.filename_maps_default)
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=maps)

        self.assertIsInstance(spaxel.cube, marvin.tools.cube.Cube)
        self.assertIsInstance(spaxel.maps, marvin.tools.maps.Maps)

        self.assertIsInstance(spaxel.spectrum, Spectrum)
        self.assertTrue(len(spaxel.properties) > 0)

    def test_cube_maps_object_filename_mpl5(self):

        marvin.config.setMPL('MPL-5')

        cube = marvin.tools.cube.Cube(filename=self.filename_cube)
        maps = marvin.tools.maps.Maps(filename=self.filename_maps_default)
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=maps)

        self.assertEqual(cube._drpver, 'v1_5_1')
        self.assertEqual(spaxel._drpver, 'v1_5_1')
        self.assertEqual(maps._drpver, 'v1_5_1')
        self.assertEqual(maps._dapver, '1.1.1')
        self.assertEqual(spaxel._dapver, '1.1.1')

        self.assertIsInstance(spaxel.cube, marvin.tools.cube.Cube)
        self.assertIsInstance(spaxel.maps, marvin.tools.maps.Maps)

        self.assertIsInstance(spaxel.spectrum, Spectrum)
        self.assertTrue(len(spaxel.properties) > 0)

    def test_cube_object_api(self):

        cube = marvin.tools.cube.Cube(plateifu=self.plateifu, mode='remote')
        spaxel = Spaxel(x=15, y=16, cube=cube)

        self.assertIsInstance(spaxel.cube, marvin.tools.cube.Cube)
        self.assertIsInstance(spaxel.maps, marvin.tools.maps.Maps)

        self.assertIsInstance(spaxel.spectrum, Spectrum)
        self.assertTrue(len(spaxel.properties) > 0)

    def test_cube_maps_object_api(self):

        cube = marvin.tools.cube.Cube(plateifu=self.plateifu, mode='remote')
        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote')
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=maps)

        self.assertIsInstance(spaxel.cube, marvin.tools.cube.Cube)
        self.assertIsInstance(spaxel.maps, marvin.tools.maps.Maps)

        self.assertIsInstance(spaxel.spectrum, Spectrum)
        self.assertTrue(len(spaxel.properties) > 0)
        self.assertIsInstance(spaxel.properties, DictOfProperties)


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
