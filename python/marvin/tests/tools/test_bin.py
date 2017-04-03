#!/usr/bin/env python
# encoding: utf-8
#
# test_bin.py
#
# Created by José Sánchez-Gallego on 6 Nov 2016.
# Modified by Brett Andrews on 3 Apr 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest

import marvin
import marvin.tests
import marvin.tools.bin
import marvin.tools.maps
import marvin.tools.modelcube
import marvin.utils.general
from marvin.core.exceptions import MarvinError


bintype = 'VOR10'


@pytest.fixture(scope='module')
def galaxy(request, init_galaxy, set_sasurl):
    galaxy = init_galaxy
    galaxy.set_filenames(bintype=bintype)
    galaxy.set_filepaths()
    galaxy.maps_filename = galaxy.mapspath
    galaxy.modelcube_filename = galaxy.modelpath
    yield galaxy


class TestBinInit:

    def _check_bin_data(self, bb, gal):

        assert bb.binid == 100
        assert bb.plateifu == gal.plateifu
        assert bb.mangaid == gal.mangaid

        assert len(bb.spaxels) == 2
        assert not bb.spaxels[0].loaded

        assert bb.properties is not None

    def test_init_from_files(self, galaxy):

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=galaxy.maps_filename,
                                  modelcube_filename=galaxy.modelcube_filename)

        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        self._check_bin_data(bb, galaxy)

    def test_init_from_file_only_maps(self, galaxy):

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=galaxy.maps_filename)

        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert bb._modelcube is not None
        assert bb._modelcube.data_origin == 'db'
        assert bb._modelcube.bintype == galaxy.bintype

        self._check_bin_data(bb, galaxy)

    def test_init_from_db(self, galaxy):

        bb = marvin.tools.bin.Bin(binid=100, plateifu=galaxy.plateifu, bintype=galaxy.bintype)
        assert bb._maps.data_origin == 'db'
        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)
        assert bb._modelcube.bintype == galaxy.bintype

        self._check_bin_data(bb, galaxy)

    def test_init_from_api(self, galaxy):

        bb = marvin.tools.bin.Bin(binid=100, plateifu=galaxy.plateifu, mode='remote',
                                  bintype=galaxy.bintype)

        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)
        assert bb._modelcube.bintype == galaxy.bintype

        self._check_bin_data(bb, galaxy)

    def test_bin_does_not_exist(self, galaxy):

        with pytest.raises(MarvinError) as ee:
            marvin.tools.bin.Bin(binid=99999, plateifu=galaxy.plateifu, mode='local',
                                 bintype=galaxy.bintype)
            assert 'there are no spaxels associated with binid=99999.' in str(ee.exception)


class TestBinFileMismatch:
    @pytest.mark.skip(reason="test doesn't work yet")
    def test_bintypes(self):

        wrong_bintype = 'SPX'
        assert wrong_bintype != self.bintype

        wrong_modelcube_filename = os.path.join(
            self.path_release,
            '{0}-GAU-MILESHC'.format(wrong_bintype), str(self.plate), str(self.ifu),
            'manga-{0}-{1}-{2}-GAU-MILESHC.fits.gz'.format(self.plateifu, 'LOGCUBE', wrong_bintype))

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=self.maps_filename,
                                  modelcube_filename=wrong_modelcube_filename)

        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        with pytest.raises(MarvinError):
            marvin.utils.general._check_file_parameters(bb._maps, bb._modelcube)()
