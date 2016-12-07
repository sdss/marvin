#!/usr/bin/env python
# encoding: utf-8
#
# test_sampledb.py
#
# Created by José Sánchez-Gallego on 6 Dec 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import marvin
import marvin.tests


class TestSampleDB(marvin.tests.MarvinTest):
    """A series of tests for the SampleModelClasses."""

    @classmethod
    def setUpClass(cls):

        marvin.config.switchSasUrl('local')

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):

        self.session = marvin.marvindb.session
        self.sampledb = marvin.marvindb.sampledb

    def tearDown(self):
        pass

    def test_elpetro_mag(self):

        expected = [18.69765903, 17.45450578, 16.80842176, 16.43652498, 16.20534984]

        nsa_target = self.session.query(
            self.sampledb.NSA).join(self.sampledb.MangaTargetToNSA,
                                    self.sampledb.MangaTarget).filter(
                self.sampledb.MangaTarget.mangaid == '1-209232').one()

        for ii, band in enumerate(['u', 'g', 'r', 'i', 'z']):
            self.assertAlmostEqual(
                getattr(nsa_target, 'elpetro_mag_{0}'.format(band)), expected[ii])

    def test_elpetro_colour(self):

        expected = [1.24315324, 0.64608403, 0.37189678, 0.23117514]
        colours = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z')]

        nsa_target = self.session.query(
            self.sampledb.NSA).join(self.sampledb.MangaTargetToNSA,
                                    self.sampledb.MangaTarget).filter(
                self.sampledb.MangaTarget.mangaid == '1-209232').one()

        for ii, bands in enumerate(colours):
            bandA, bandB = bands
            self.assertAlmostEqual(nsa_target.elpetro_colour(bandA, bandB), expected[ii])
            self.assertAlmostEqual(getattr(nsa_target,
                                           'elpetro_mag_{0}_{1}'.format(bandA, bandB)),
                                   expected[ii])
