# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-12 18:41:25
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-06-12 18:58:55

from __future__ import print_function, division, absolute_import
import pytest


@pytest.fixture()
def nsa_target(maindb, galaxy):
    nsa = maindb.session.query(maindb.sampledb.NSA).join(maindb.sampledb.MangaTargetToNSA,
                                                         maindb.sampledb.MangaTarget).\
        filter(maindb.sampledb.MangaTarget.mangaid == galaxy.mangaid).one()
    yield nsa
    nsa = None


class TestSampleDB(object):

    @pytest.mark.parametrize('band,expected', [('u', 18.69765903),
                                               ('g', 17.45450578),
                                               ('r', 16.80842176),
                                               ('i', 16.43652498),
                                               ('z', 16.20534984)])
    def test_elpetro_mag(self, nsa_target, band, expected):

        assert pytest.approx(getattr(nsa_target, 'elpetro_mag_{0}'.format(band)), expected)

    @pytest.mark.parametrize('bands,expected', [(('u', 'g'), 1.24315324),
                                                (('g', 'r'), 0.64608403),
                                                (('r', 'i'), 0.37189678),
                                                (('i', 'z'), 0.23117514)])
    def test_elpetro_colour(self, nsa_target, bands, expected):

        bandA, bandB = bands
        assert pytest.approx(nsa_target.elpetro_colour(bandA, bandB), expected)
        elpetro_mag_colour = getattr(nsa_target,
                                     'elpetro_mag_{0}_{1}'.format(bandA, bandB))
        assert pytest.approx(elpetro_mag_colour, expected)

    @pytest.mark.parametrize('bands,expected', [(('u', 'g'), 1.1655902862549006),
                                                (('g', 'r'), 0.5961246490479013),
                                                (('r', 'i'), 0.3375816345214986),
                                                (('i', 'z'), 0.20068740844720168)])
    def test_elpetro_absmag_colour(self, nsa_target, bands, expected):

        bandA, bandB = bands
        assert pytest.approx(nsa_target.elpetro_absmag_colour(bandA, bandB), expected)
        elpetro_absmag_colour = getattr(nsa_target, 'elpetro_absmag_{0}_{1}'.format(bandA, bandB))
        assert pytest.approx(elpetro_absmag_colour, expected)

