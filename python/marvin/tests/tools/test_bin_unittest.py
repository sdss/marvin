#!/usr/bin/env python
# encoding: utf-8
#
# test_bin.py
#
# Created by José Sánchez-Gallego on 6 Nov 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import unittest

import marvin
import marvin.tests
import marvin.tools.bin
import marvin.tools.maps
import marvin.tools.modelcube
import marvin.utils.general
from marvin.core.exceptions import MarvinError


class TestBinBase(marvin.tests.MarvinTest):
    """Defines the files and plateifus we will use in the tests."""

    @classmethod
    def setUpClass(cls):
        super(TestBinBase, cls).setUpClass()
        cls._update_release('MPL-5')
        cls.set_sasurl('local')
        cls.set_filepaths(bintype='VOR10')
        cls.maps_filename = cls.mapspath
        cls.modelcube_filename = cls.modelpath

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self._reset_the_config()
        self._update_release('MPL-5')
        self.set_sasurl('local')
        self.assertTrue(os.path.exists(self.maps_filename))
        self.assertTrue(os.path.exists(self.modelcube_filename))

    def tearDown(self):
        pass


class TestBinInit(TestBinBase):

    def _check_bin_data(self, bb):

        self.assertEqual(bb.binid, 100)
        self.assertEqual(bb.plateifu, self.plateifu)
        self.assertEqual(bb.mangaid, self.mangaid)

        self.assertTrue(len(bb.spaxels) == 2)
        self.assertFalse(bb.spaxels[0].loaded)

        self.assertIsNotNone(bb.properties)

    def test_init_from_files(self):

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=self.maps_filename,
                                  modelcube_filename=self.modelcube_filename)

        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsInstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        self._check_bin_data(bb)

    def test_init_from_file_only_maps(self):

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=self.maps_filename)

        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsNotNone(bb._modelcube)
        self.assertEqual(bb._modelcube.data_origin, 'db')
        self.assertEqual(bb._modelcube.bintype, self.bintype)

        self._check_bin_data(bb)

    def test_init_from_db(self):

        bb = marvin.tools.bin.Bin(binid=100, plateifu=self.plateifu, bintype=self.bintype)
        self.assertEqual(bb._maps.data_origin, 'db')
        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsInstance(bb._modelcube, marvin.tools.modelcube.ModelCube)
        self.assertEqual(bb._modelcube.bintype, self.bintype)

        self._check_bin_data(bb)

    def test_init_from_api(self):

        bb = marvin.tools.bin.Bin(binid=100, plateifu=self.plateifu, mode='remote',
                                  bintype=self.bintype)

        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsInstance(bb._modelcube, marvin.tools.modelcube.ModelCube)
        self.assertEqual(bb._modelcube.bintype, self.bintype)

        self._check_bin_data(bb)

    def test_bin_does_not_exist(self):

        with self.assertRaises(MarvinError) as ee:
            marvin.tools.bin.Bin(binid=99999, plateifu=self.plateifu, mode='local',
                                 bintype=self.bintype)
            self.assertIn('there are no spaxels associated with binid=99999.', str(ee.exception))


class TestBinFileMismatch(TestBinBase):

    @unittest.expectedFailure
    def test_bintypes(self):

        wrong_bintype = 'SPX'
        self.assertNotEqual(wrong_bintype, self.bintype)

        wrong_modelcube_filename = os.path.join(
            self.path_release,
            '{0}-GAU-MILESHC'.format(wrong_bintype), str(self.plate), str(self.ifu),
            'manga-{0}-{1}-{2}-GAU-MILESHC.fits.gz'.format(self.plateifu, 'LOGCUBE', wrong_bintype))

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=self.maps_filename,
                                  modelcube_filename=wrong_modelcube_filename)

        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsInstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        self.assertRaises(MarvinError,
                          marvin.utils.general._check_file_parameters(bb._maps, bb._modelcube))


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
