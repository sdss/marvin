#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Nov 1, 2017
# @Filename: test_spaxel.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
import os

import pytest
import astropy.io.fits

from marvin import config

from marvin.core.exceptions import MarvinError

from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube
from marvin.tools.spaxel import SpaxelBase


spaxel_modes = [True, False, 'object']


@pytest.fixture(params=itertools.product(spaxel_modes, spaxel_modes, spaxel_modes))
def cube_maps_modelcube_modes(request):
    return request.param


@pytest.fixture
def galaxy_spaxel(galaxy):
    """Returns only some instances of the galaxy fixture."""

    if galaxy.release == 'MPL-4':
        if galaxy.template in ['M11-STELIB-ZSOL', 'MILES-THIN']:
            pytest.skip()
        if galaxy.bintype == 'RADIAL':
            pytest.skip()

    if galaxy.release == 'MPL-4':
        if galaxy.bintype in ['NRE', 'ALL']:
            pytest.skip()

    return galaxy


class TestSpaxel(object):

    def test_SpaxelBase(self, galaxy_spaxel, cube_maps_modelcube_modes):

        plateifu = galaxy_spaxel.plateifu
        bintype = galaxy_spaxel.bintype.name
        template = galaxy_spaxel.template.name
        release = galaxy_spaxel.release

        cube, maps, modelcube = cube_maps_modelcube_modes
        cube_filename = maps_filename = modelcube_filename = None

        if cube == 'object':
            cube = Cube(plateifu=plateifu, release=release)

        if maps == 'object':
            maps = Maps(plateifu=plateifu, bintype=bintype,
                        template=template, release=release)

        if release == 'MPL-4':
            modelcube = False
        elif modelcube == 'object':
            modelcube = ModelCube(plateifu=plateifu, bintype=bintype,
                                  template=template, release=release)

        if cube is False and maps is False and modelcube is False:
            pytest.skip()

        spaxel = SpaxelBase(x=15, y=15, plateifu=plateifu,
                            cube=cube, maps=maps, modelcube=modelcube,
                            cube_filename=cube_filename,
                            maps_filename=maps_filename,
                            modelcube_filename=modelcube_filename,
                            template=template, bintype=bintype)

        assert isinstance(spaxel, SpaxelBase)

    def test_no_inputs(self):

        with pytest.raises(MarvinError) as ee:
            SpaxelBase(x=0, y=0, cube=None, maps=None, modelcube=None)

        assert 'either cube, maps, or modelcube must be' in str(ee)


class TestPickling(object):

    def test_pickling_db_fails(self, temp_scratch, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        spaxel = cube.getSpaxel(x=1, y=3)

        file = temp_scratch.join('test_spaxel.mpf')

        with pytest.raises(MarvinError) as cm:
            spaxel.save(str(file), overwrite=True)

        assert 'objects with data_origin=\'db\' cannot be saved.' in str(cm.value)

    def test_pickling_only_cube_file(self, temp_scratch, galaxy):
        if galaxy.bintype.name != 'SPX':
            pytest.skip("Can't instantiate a Spaxel from a binned Maps.")

        cube = Cube(filename=galaxy.cubepath)
        maps = Maps(filename=galaxy.mapspath)

        spaxel = cube.getSpaxel(x=1, y=3, properties=maps, models=False)

        file = temp_scratch.join('test_spaxel.mpf')

        path_saved = spaxel.save(str(file), overwrite=True)
        assert file.check() is True
        assert os.path.exists(path_saved)

        del spaxel

        spaxel_restored = SpaxelBase.restore(str(file))
        assert spaxel_restored is not None
        assert isinstance(spaxel_restored, SpaxelBase)

        assert spaxel_restored.cube is not None
        assert spaxel_restored.cube.data_origin == 'file'
        assert isinstance(spaxel_restored.cube.data, astropy.io.fits.HDUList)

        assert spaxel_restored.maps is not None
        assert spaxel_restored.maps.data_origin == 'file'
        assert isinstance(spaxel_restored.maps.data, astropy.io.fits.HDUList)

    @pytest.mark.parametrize('mpl', ['MPL-5'])
    def test_pickling_all_api(self, monkeypatch, temp_scratch, galaxy, mpl):
        monkeypatch.setattr(config, 'release', mpl)
        drpver, __ = config.lookUpVersions()

        cube = Cube(plateifu=galaxy.plateifu, mode='remote')
        maps = Maps(plateifu=galaxy.plateifu, mode='remote')
        modelcube = ModelCube(plateifu=galaxy.plateifu, mode='remote')
        spaxel = cube.getSpaxel(x=1, y=3, properties=maps, models=modelcube)

        assert spaxel.cube.data_origin == 'api'
        assert spaxel.maps.data_origin == 'api'
        assert spaxel.modelcube.data_origin == 'api'

        file = temp_scratch.join('test_spaxel_api.mpf')

        path_saved = spaxel.save(str(file), overwrite=True)
        assert file.check() is True
        assert os.path.exists(path_saved)

        del spaxel

        spaxel_restored = SpaxelBase.restore(str(file))
        assert spaxel_restored is not None
        assert isinstance(spaxel_restored, SpaxelBase)

        assert spaxel_restored.cube is not None
        assert isinstance(spaxel_restored.cube, Cube)
        assert spaxel_restored.cube.data_origin == 'api'
        assert spaxel_restored.cube.data is None
        assert spaxel_restored.cube.header['VERSDRP3'] == drpver

        assert spaxel_restored.maps is not None
        assert isinstance(spaxel_restored.maps, Maps)
        assert spaxel_restored.maps.data_origin == 'api'
        assert spaxel_restored.maps.data is None

        assert spaxel_restored.modelcube is not None
        assert isinstance(spaxel_restored.modelcube, ModelCube)
        assert spaxel_restored.modelcube.data_origin == 'api'
        assert spaxel_restored.modelcube.data is None
