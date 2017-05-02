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

import copy
import os
import unittest

import astropy.io.fits
import numpy as np

import marvin
import marvin.tests
import marvin.tools.map
import marvin.tools.maps
import marvin.tools.spaxel

from marvin.tests import UseBintypes


def _assert_maps(maps, galaxy):
    """Basic checks for a Maps object."""

    assert maps is not None
    assert maps.plateifu == galaxy.plateifu
    assert maps.mangaid == galaxy.mangaid
    assert maps.wcs is not None
    assert maps.bintype == galaxy.bintype
    assert all([xx == 34 for xx in list(maps.shape)])


@UseBintypes('VOR10')
class TestMapsFile(object):

    def test_load_from_file(self, galaxy):

        maps = marvin.tools.maps.Maps(filename=galaxy.maps_filename)
        _assert_maps(maps, galaxy)
        assert maps.data is not None
        assert isinstance(maps.data, astropy.io.fits.HDUList)
        assert maps.cube is not None
        assert maps.cube.plateifu == galaxy.plateifu
        assert maps.cube.mangaid == galaxy.mangaid

    def test_load_file_mpl4_global_mpl5(self, galaxy):

        marvin.config.setMPL('MPL-5')
        maps = marvin.tools.maps.Maps(filename=galaxy.maps_filename)
        assert maps._release == 'MPL-4'
        assert maps._drpver == 'v1_5_1'
        assert maps._dapver == '1.1.1'

    # def test_get_spaxel_file(self, galaxy):
    #
    #     maps = marvin.tools.maps.Maps(filename=self.filename_default)
    #     spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower')
    #
    #     self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
    #     self.assertIsNotNone(spaxel.spectrum)
    #     self.assertTrue(len(spaxel.properties.keys()) > 0)
    #
    #     self.assertAlmostEqual(spaxel.properties['stellar_vel'].ivar, 1.013657e-05)
    #
    # def test_get_spaxel_test2_file(self, galaxy):
    #
    #     maps = marvin.tools.maps.Maps(filename=self.filename_default)
    #     spaxel = maps.getSpaxel(x=5, y=5)
    #
    #     self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
    #     self.assertIsNotNone(spaxel.spectrum)
    #     self.assertTrue(len(spaxel.properties.keys()) > 0)
    #
    # def test_get_spaxel_no_db(self, galaxy):
    #     """Tests getting an spaxel if there is no DB."""
    #
    #     marvin.marvindb.db = None
    #
    #     maps = marvin.tools.maps.Maps(filename=self.filename_default)
    #     spaxel = maps.getSpaxel(x=5, y=5)
    #
    #     self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
    #     self.assertIsNotNone(spaxel.spectrum)
    #     self.assertTrue(len(spaxel.properties.keys()) > 0)
    #
    # def test_get_spaxel_binned_maps(self, galaxy):
    #
    #     maps = marvin.tools.maps.Maps(plateifu=self.plateifu, bintype='VOR10',
    #                                   release='MPL-5')
    #     spaxel = maps.getSpaxel(x=15, y=8, xyorig='lower')
    #
    #     self.assertTrue(isinstance(spaxel, marvin.tools.spaxel.Spaxel))
    #     self.assertIsNotNone(spaxel.spectrum)
    #     self.assertTrue(len(spaxel.properties.keys()) > 0)
    #
    #     self.assertAlmostEqual(spaxel.properties['stellar_vel'].ivar, 0.00031520479546875247)
    #     self.assertEqual(spaxel.bintype, 'SPX')
    #
    # def test_bintype_maps_filename_vor10(self, galaxy):
    #
    #     maps = marvin.tools.maps.Maps(filename=self.filename_mpl5_vor10, release='MPL-5')
    #     self.assertEqual(maps.bintype, 'VOR10')
    #
    # def test_bintype_maps_filename_spx(self, galaxy):
    #     maps = marvin.tools.maps.Maps(filename=self.filename_mpl5_spx, release='MPL-5')
    #     self.assertEqual(maps.bintype, 'SPX')
    #
    # def test_bintype_maps_filename_bad_input(self, galaxy):
    #
    #     maps = marvin.tools.maps.Maps(filename=self.filename_mpl5_spx, bintype='VOR10',
    #                                   release='MPL-5')
    #     self.assertEqual(maps.bintype, 'SPX')
