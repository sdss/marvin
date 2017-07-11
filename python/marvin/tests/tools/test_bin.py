#!/usr/bin/env python
# encoding: utf-8
#
# test_bin.py
#
# Created by José Sánchez-Gallego on 6 Nov 2016.
# Modified by Brett Andrews on 24 Apr 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import pytest

import marvin
import marvin.tools.maps
import marvin.tools.modelcube
import marvin.utils.general
import marvin.tests
from marvin.tools.bin import Bin
from marvin.tests import UseBintypes, marvin_test_if_class
from marvin.core.exceptions import MarvinError


@UseBintypes('VOR10')
@pytest.mark.xfail(run=False, reason='expected fail until someone fixes it')
class TestBinInit:

    def _check_bin_data(self, bb, gal):

        assert bb.binid == 100
        assert bb.plateifu == gal.plateifu
        assert bb.mangaid == gal.mangaid

        assert len(bb.spaxels) == 2
        assert not bb.spaxels[0].loaded

        assert bb.properties is not None

    def test_init_from_files(self, galaxy):

        bb = Bin(binid=100, maps_filename=galaxy.mapspath, modelcube_filename=galaxy.modelpath)

        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        self._check_bin_data(bb, galaxy)

    def test_init_from_file_only_maps(self, galaxy):

        bb = Bin(binid=100, maps_filename=galaxy.mapspath)

        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert bb._modelcube is not None
        assert bb._modelcube.data_origin == 'db'
        assert bb._modelcube.bintype == galaxy.bintype

        self._check_bin_data(bb, galaxy)

    def test_init_from_db(self, galaxy):

        bb = Bin(binid=100, plateifu=galaxy.plateifu, bintype=galaxy.bintype)
        assert bb._maps.data_origin == 'db'
        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)
        assert bb._modelcube.bintype == galaxy.bintype

        self._check_bin_data(bb, galaxy)

    def test_init_from_api(self, galaxy):

        bb = Bin(binid=100, plateifu=galaxy.plateifu, mode='remote', bintype=galaxy.bintype)

        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)
        assert bb._modelcube.bintype == galaxy.bintype

        self._check_bin_data(bb, galaxy)

    def test_bin_does_not_exist(self, galaxy):

        with pytest.raises(MarvinError) as ee:
            Bin(binid=99999, plateifu=galaxy.plateifu, mode='local', bintype=galaxy.bintype)
            assert 'there are no spaxels associated with binid=99999.' in str(ee.exception)


#@marvin_test_if_class(mark='skip', galaxy=dict(release=['MPL-4']))
@pytest.mark.xfail(run=False, reason='expected fail until someone fixes it')
class TestBinFileMismatch:
    def test_bintypes(self, galaxy):

        wrong_bintype = 'WRONGSPX'
        assert wrong_bintype != galaxy.bintype

        wrong_modelcube_filename = galaxy.new_path('modelcube', {'bintype': wrong_bintype, 'daptype': '{0}-GAU-MILESHC'.format(wrong_bintype)})

        with pytest.raises(MarvinError) as cm:
            bb = Bin(binid=100, maps_filename=galaxy.mapspath, modelcube_filename=wrong_modelcube_filename)

        assert 'there are no spaxels associated with binid=100.' in str(cm.value)

        # assert isinstance(bb._maps, marvin.tools.maps.Maps)
        # assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        # with pytest.raises(MarvinError):
        #     marvin.utils.general._check_file_parameters(bb._maps, bb._modelcube)()
