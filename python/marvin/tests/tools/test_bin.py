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

import marvin
import marvin.tests
import marvin.tools.bin
import marvin.tools.maps
import marvin.tools.modelcube


class TestBinBase(marvin.tests.MarvinTest):
    """Defines the files and plateifus we will use in the tests."""

    @classmethod
    def setUpClass(cls):

        marvin.config.switchSasUrl('local')

        cls.drpver = 'v2_0_1'
        cls.dapver = '2.0.2'
        cls.bintype = 'VOR10'

        cls.plate = 8485
        cls.mangaid = '1-209232'
        cls.plateifu = '8485-1901'
        cls.ifu = cls.plateifu.split('-')[1]

        cls.maps_filename = os.path.join(
            os.getenv('MANGA_SPECTRO_ANALYSIS'), cls.drpver, cls.dapver,
            '{0}-GAU-MILESHC'.format(cls.bintype), str(cls.plate), str(cls.ifu),
            'manga-{0}-{1}-{2}-GAU-MILESHC.fits.gz'.format(cls.plateifu, 'MAPS', cls.bintype))

        cls.modelcube_filename = os.path.join(
            os.getenv('MANGA_SPECTRO_ANALYSIS'), cls.drpver, cls.dapver,
            '{0}-GAU-MILESHC'.format(cls.bintype), str(cls.plate), str(cls.ifu),
            'manga-{0}-{1}-{2}-GAU-MILESHC.fits.gz'.format(cls.plateifu, 'LOGCUBE', cls.bintype))

        cls.marvindb_session = marvin.marvindb.session

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):

        marvin.marvindb.session = self.marvindb_session
        marvin.config.setMPL('MPL-5')

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
        self.assertFalse(bb._modelcube)

        self._check_bin_data(bb)

    def test_init_from_db(self):

        bb = marvin.tools.bin.Bin(binid=100, plateifu=self.plateifu, bintype=self.bintype)
        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsInstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        self._check_bin_data(bb)

    def test_init_from_api(self):

        bb = marvin.tools.bin.Bin(binid=100, plateifu=self.plateifu, mode='remote',
                                  bintype=self.bintype)

        self.assertIsInstance(bb._maps, marvin.tools.maps.Maps)
        self.assertIsInstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        self._check_bin_data(bb)
