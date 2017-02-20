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

import astropy.io.fits

import marvin
import marvin.tests
import marvin.tools.cube
import marvin.tools.maps

from marvin.core.exceptions import MarvinError
from marvin.tools.analysis_props import DictOfProperties
from marvin.tools.spaxel import Spaxel
from marvin.tools.spectrum import Spectrum


class TestSpaxelBase(marvin.tests.MarvinTest):
    """Defines the files and plateifus we will use in the tests."""

    @classmethod
    def setUpClass(cls):

        super(TestSpaxelBase, cls).setUpClass()
        cls.set_sasurl('local')
        cls._update_release('MPL-4')
        cls.set_filepaths()

        cls.filename_cube = os.path.realpath(cls.cubepath)
        cls.filename_maps_default = os.path.join(
            cls.mangaanalysis, cls.drpver, cls.dapver,
            'default', str(cls.plate), 'mangadap-{0}-default.fits.gz'.format(cls.plateifu))

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self._update_release('MPL-4')
        marvin.config.forceDbOn()

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

    def test_db_maps_miles(self):

        spaxel = Spaxel(x=15, y=16, cube=False, modelcube=False, maps=True,
                        plateifu=self.plateifu,
                        template_kin='MILES-THIN')
        self.assertEqual(spaxel.maps.template_kin, 'MILES-THIN')

    def test_api_maps_invalid_template(self):

        with self.assertRaises(AssertionError) as cm:
            Spaxel(x=15, y=16, cube=False, modelcube=False, maps=True,
                   plateifu=self.plateifu,
                   template_kin='MILES-TH')
        self.assertIn('invalid template_kin', str(cm.exception))

    def test_load_false(self):

        spaxel = Spaxel(plateifu=self.plateifu, x=15, y=16, load=False)

        self.assertFalse(spaxel.loaded)
        self.assertTrue(spaxel.cube)
        self.assertTrue(spaxel.maps)
        self.assertTrue(spaxel.modelcube)
        self.assertEqual(len(spaxel.properties), 0)
        self.assertIsNone(spaxel.spectrum)

        spaxel.load()

        self.assertIsInstance(spaxel.cube, marvin.tools.cube.Cube)
        self.assertIsInstance(spaxel.maps, marvin.tools.maps.Maps)

        self.assertIsInstance(spaxel.spectrum, Spectrum)
        self.assertTrue(len(spaxel.properties) > 0)
        self.assertIsInstance(spaxel.properties, DictOfProperties)

    def test_fails_unbinned_maps(self):

        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, bintype='VOR10',
                                      release='MPL-5')

        with self.assertRaises(MarvinError) as cm:
            Spaxel(x=15, y=16, plateifu=self.plateifu, maps=maps)

        self.assertIn('cannot instantiate a Spaxel from a binned Maps.', str(cm.exception))

    def test_spaxel_ra_dec(self):

        cube = marvin.tools.cube.Cube(plateifu=self.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube)

        self.assertAlmostEqual(spaxel.ra, 232.54512, places=5)
        self.assertAlmostEqual(spaxel.dec, 48.690062, places=5)

    def test_release(self):

        cube = marvin.tools.cube.Cube(plateifu=self.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube)

        self.assertEqual(spaxel.release, 'MPL-4')

        with self.assertRaises(MarvinError) as ee:
            spaxel.release = 'a'
            self.assertIn('the release cannot be changed', str(ee.exception))


class TestPickling(TestSpaxelBase):

    def setUp(self):
        super(TestPickling, self).setUp()
        self._files_created = []

    def tearDown(self):

        super(TestPickling, self).tearDown()

        for fp in self._files_created:
            full_fp = os.path.realpath(os.path.expanduser(fp))
            if os.path.exists(full_fp):
                os.remove(full_fp)

    def test_pickling_db_fails(self):

        cube = marvin.tools.cube.Cube(plateifu=self.plateifu)
        spaxel = cube.getSpaxel(x=1, y=3)

        spaxel_path = '~/test_spaxel.mpf'
        self._files_created.append(spaxel_path)
        with self.assertRaises(MarvinError) as ee:
            spaxel.save(spaxel_path, overwrite=True)

        self.assertIn('objects with data_origin=\'db\' cannot be saved.', str(ee.exception))

    def test_pickling_only_cube_file(self):

        cube = marvin.tools.cube.Cube(filename=self.filename_cube)
        maps = marvin.tools.maps.Maps(filename=self.filename_maps_default)

        spaxel = cube.getSpaxel(x=1, y=3, properties=maps, modelcube=False)

        spaxel_path = '~/test_spaxel.mpf'
        self._files_created.append(spaxel_path)

        path_saved = spaxel.save(spaxel_path, overwrite=True)
        self.assertTrue(os.path.exists(path_saved))
        self.assertTrue(os.path.realpath(os.path.expanduser(spaxel_path)), path_saved)

        del spaxel

        spaxel_restored = marvin.tools.spaxel.Spaxel.restore(spaxel_path)
        self.assertIsNotNone(spaxel_restored)
        self.assertIsInstance(spaxel_restored, marvin.tools.spaxel.Spaxel)

        self.assertIsNotNone(spaxel_restored.cube)
        self.assertTrue(spaxel_restored.cube.data_origin == 'file')
        self.assertIsInstance(spaxel_restored.cube.data, astropy.io.fits.HDUList)

        self.assertIsNotNone(spaxel_restored.maps)
        self.assertTrue(spaxel_restored.maps.data_origin == 'file')
        self.assertIsInstance(spaxel_restored.maps.data, astropy.io.fits.HDUList)

    def test_pickling_all_api(self):

        marvin.config.marvindb = None
        marvin.config.switchSasUrl('local')
        marvin.config.setMPL('MPL-5')

        cube = marvin.tools.cube.Cube(plateifu=self.plateifu, mode='remote')
        maps = marvin.tools.maps.Maps(plateifu=self.plateifu, mode='remote')
        modelcube = marvin.tools.modelcube.ModelCube(plateifu=self.plateifu, mode='remote')

        spaxel = cube.getSpaxel(x=1, y=3, properties=maps, modelcube=modelcube)

        self.assertEqual(spaxel.cube.data_origin, 'api')
        self.assertEqual(spaxel.maps.data_origin, 'api')
        self.assertEqual(spaxel.modelcube.data_origin, 'api')

        spaxel_path = '~/test_spaxel_api.mpf'
        self._files_created.append(spaxel_path)

        path_saved = spaxel.save(spaxel_path, overwrite=True)
        self.assertTrue(os.path.exists(path_saved))
        self.assertTrue(os.path.realpath(os.path.expanduser(spaxel_path)), path_saved)

        del spaxel

        spaxel_restored = marvin.tools.spaxel.Spaxel.restore(spaxel_path)
        self.assertIsNotNone(spaxel_restored)
        self.assertIsInstance(spaxel_restored, marvin.tools.spaxel.Spaxel)

        self.assertIsNotNone(spaxel_restored.cube)
        self.assertIsInstance(spaxel_restored.cube, marvin.tools.cube.Cube)
        self.assertTrue(spaxel_restored.cube.data_origin == 'api')
        self.assertIsNone(spaxel_restored.cube.data)
        self.assertEqual(spaxel_restored.cube.header['VERSDRP3'], 'v2_0_1')

        self.assertIsNotNone(spaxel_restored.maps)
        self.assertIsInstance(spaxel_restored.maps, marvin.tools.maps.Maps)
        self.assertTrue(spaxel_restored.maps.data_origin == 'api')
        self.assertIsNone(spaxel_restored.maps.data)

        self.assertIsNotNone(spaxel_restored.modelcube)
        self.assertIsInstance(spaxel_restored.modelcube, marvin.tools.modelcube.ModelCube)
        self.assertTrue(spaxel_restored.modelcube.data_origin == 'api')
        self.assertIsNone(spaxel_restored.modelcube.data)


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
