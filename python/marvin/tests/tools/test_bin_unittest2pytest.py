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
import pytest

import marvin
import marvin.tests
import marvin.tools.bin
import marvin.tools.maps
import marvin.tools.modelcube
import marvin.utils.general
from marvin.core.exceptions import MarvinError

params = [('8485-1901', 'MPL-5')]


@pytest.fixture(scope='module', params=params)
def set_galaxy(request, galaxy):
    def fin():
        pass
    request.addfinalizer(fin)


# TODO replace with module level fixture
# TODO call Galaxy class
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
        assert os.path.exists(self.maps_filename)
        assert os.path.exists(self.modelcube_filename)

    def tearDown(self):
        pass


class TestBinInit:

    def _check_bin_data(self, bb):

        assert bb.binid == 100
        assert bb.plateifu == self.plateifu
        assert bb.mangaid == self.mangaid

        assert len(bb.spaxels) == 2
        assert not bb.spaxels[0].loaded

        assert bb.properties is not None

    def test_init_from_files(self, set_galaxy):

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=self.maps_filename,
                                  modelcube_filename=self.modelcube_filename)

        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)

        self._check_bin_data(bb)

    def test_init_from_file_only_maps(self):

        bb = marvin.tools.bin.Bin(binid=100, maps_filename=self.maps_filename)

        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert bb._modelcube is not None
        assert bb._modelcube.data_origin == 'db'
        assert bb._modelcube.bintype == self.bintype

        self._check_bin_data(bb)

    def test_init_from_db(self):

        bb = marvin.tools.bin.Bin(binid=100, plateifu=self.plateifu, bintype=self.bintype)
        assert bb._maps.data_origin == 'db'
        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)
        assert bb._modelcube.bintype == self.bintype

        self._check_bin_data(bb)

    def test_init_from_api(self):

        bb = marvin.tools.bin.Bin(binid=100, plateifu=self.plateifu, mode='remote',
                                  bintype=self.bintype)

        assert isinstance(bb._maps, marvin.tools.maps.Maps)
        assert isinstance(bb._modelcube, marvin.tools.modelcube.ModelCube)
        assert bb._modelcube.bintype == self.bintype

        self._check_bin_data(bb)

    def test_bin_does_not_exist(self):

        with pytest.raises(MarvinError) as ee:
            marvin.tools.bin.Bin(binid=99999, plateifu=self.plateifu, mode='local',
                                 bintype=self.bintype)
            assert 'there are no spaxels associated with binid=99999.' in str(ee.exception)


class TestBinFileMismatch(TestBinBase):
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
