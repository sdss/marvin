# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-12 18:41:25
# @Last modified by:   andrews
# @Last modified time: 2018-03-02 17:03:81

from __future__ import print_function, division, absolute_import
import pytest

pytestmark = pytest.mark.uses_db


@pytest.fixture()
def nsa_target(maindb, galaxy):
    nsa = maindb.session.query(maindb.sampledb.NSA).join(maindb.sampledb.MangaTargetToNSA,
                                                         maindb.sampledb.MangaTarget).\
        filter(maindb.sampledb.MangaTarget.mangaid == galaxy.mangaid).one()
    yield (nsa, galaxy.plateifu)
    nsa = None


class TestSampleDB(object):

    @pytest.mark.parametrize('plateifu, expected',
                             [('8485-1901', {'u': 18.69765903,
                                             'g': 17.45450578,
                                             'r': 16.80842176,
                                             'i': 16.43652498,
                                             'z': 16.20534984}),
                              ('7443-12701', {'u': 17.07501837,
                                              'g': 15.57770095,
                                              'r': 14.95969099,
                                              'i': 14.63861064,
                                              'z': 14.44369601})])
    def test_elpetro_mag(self, nsa_target, plateifu, expected):
        nsa_target, galaxy_plateifu = nsa_target

        if galaxy_plateifu != plateifu:
            pytest.skip('Skip non-matching plateifus.')

        for band, value in expected.items():
            assert getattr(nsa_target, 'elpetro_mag_{0}'.format(band)) == pytest.approx(value)

    @pytest.mark.parametrize('plateifu, expected',
                             [('8485-1901', {('u', 'g'): 1.24315324,
                                             ('g', 'r'): 0.64608403,
                                             ('r', 'i'): 0.37189678,
                                             ('i', 'z'): 0.23117514}),
                              ('7443-12701', {('u', 'g'): 1.49731742,
                                              ('g', 'r'): 0.61800996,
                                              ('r', 'i'): 0.32108036,
                                              ('i', 'z'): 0.19491463})])
    def test_elpetro_colour(self, nsa_target, plateifu, expected):
        nsa_target, galaxy_plateifu = nsa_target

        if galaxy_plateifu != plateifu:
            pytest.skip('Skip non-matching plateifus.')

        for bands, value in expected.items():
            bandA, bandB = bands

            assert nsa_target.elpetro_colour(bandA, bandB) == pytest.approx(value)

            elpetro_mag_colour = getattr(nsa_target, 'elpetro_mag_{0}_{1}'.format(bandA, bandB))
            assert elpetro_mag_colour == pytest.approx(value)

    @pytest.mark.parametrize('plateifu, expected',
                             [('8485-1901', {('u', 'g'): 1.1655902862549006,
                                             ('g', 'r'): 0.5961246490479013,
                                             ('r', 'i'): 0.3375816345214986,
                                             ('i', 'z'): 0.20068740844720168}),
                              ('7443-12701', {('u', 'g'): 1.3728961944580007,
                                              ('g', 'r'): 0.5836753845213991,
                                              ('r', 'i'): 0.27035522460939987,
                                              ('i', 'z'): 0.1656112670899006})])
    def test_elpetro_absmag_colour(self, nsa_target, plateifu, expected):
        nsa_target, galaxy_plateifu = nsa_target

        if galaxy_plateifu != plateifu:
            pytest.skip('Skip non-matching plateifus.')

        for bands, value in expected.items():
            bandA, bandB = bands

            assert nsa_target.elpetro_absmag_colour(bandA, bandB) == pytest.approx(value)

            colour = 'elpetro_absmag_{0}_{1}'.format(bandA, bandB)
            elpetro_absmag_colour = getattr(nsa_target, colour)
            assert elpetro_absmag_colour == pytest.approx(value)
