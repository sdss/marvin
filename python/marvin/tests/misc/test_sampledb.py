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

import pytest


class TestSampleDB(object):
    """A series of tests for the SampleModelClasses."""

    @classmethod
    def setup_class(cls):
        marvin.config.switchSasUrl('local')
        cls.session = marvin.marvindb.session
        cls.sampledb = marvin.marvindb.sampledb
        cls.nsa_target = cls.session.query(
            cls.sampledb.NSA).join(cls.sampledb.MangaTargetToNSA,
                                   cls.sampledb.MangaTarget).filter(
                cls.sampledb.MangaTarget.mangaid == '1-209232').one()

    @pytest.mark.parametrize('band,expected', [('u', 18.69765903),
                                               ('g', 17.45450578),
                                               ('r', 16.80842176),
                                               ('i', 16.43652498),
                                               ('z', 16.20534984)])
    def test_elpetro_mag(self, band, expected):

        assert pytest.approx(getattr(self.nsa_target, 'elpetro_mag_{0}'.format(band)), expected)

    @pytest.mark.parametrize('bands,expected', [(('u', 'g'), 1.24315324),
                                                (('g', 'r'), 0.64608403),
                                                (('r', 'i'), 0.37189678),
                                                (('i', 'z'), 0.23117514)])
    def test_elpetro_colour(self, bands, expected):

        bandA, bandB = bands
        assert pytest.approx(self.nsa_target.elpetro_colour(bandA, bandB), expected)
        elpetro_mag_colour = getattr(self.nsa_target,
                                     'elpetro_mag_{0}_{1}'.format(bandA, bandB))
        assert pytest.approx(elpetro_mag_colour, expected)

    def test_query_elpetro_mag(self):

        elpetro_mag_g = self.session.query(self.sampledb.NSA.elpetro_mag_g).join(
            self.sampledb.MangaTargetToNSA, self.sampledb.MangaTarget).filter(
                self.sampledb.MangaTarget.mangaid == '1-209232').first()

        assert pytest.approx(elpetro_mag_g[0], 17.454505782813705)

    @pytest.mark.parametrize('bands,expected', [(('u', 'g'), 1.1655902862549006),
                                                (('g', 'r'), 0.5961246490479013),
                                                (('r', 'i'), 0.3375816345214986),
                                                (('i', 'z'), 0.20068740844720168)])
    def test_elpetro_absmag_colour(self, bands, expected):

        bandA, bandB = bands
        assert pytest.approx(self.nsa_target.elpetro_absmag_colour(bandA, bandB), expected)
        elpetro_absmag_colour = getattr(self.nsa_target,
                                        'elpetro_absmag_{0}_{1}'.format(bandA, bandB))
        assert pytest.approx(elpetro_absmag_colour, expected)
