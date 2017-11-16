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
from marvin.tools.spaxel import SpaxelBase, Spaxel, Bin


spaxel_modes = [True, False, 'object']


@pytest.fixture(params=itertools.product(spaxel_modes, spaxel_modes, spaxel_modes))
def cube_maps_modelcube_modes(request):
    return request.param


@pytest.fixture
def galaxy_spaxel(galaxy):
    """Returns only some instances of the galaxy fixture."""

    if galaxy.plateifu != '8485-1901':
        pytest.skip()

    if galaxy.release == 'MPL-4':
        if galaxy.template in ['M11-STELIB-ZSOL', 'MILES-THIN']:
            pytest.skip()
        if galaxy.bintype == 'RADIAL':
            pytest.skip()

    if galaxy.release in ['MPL-5', 'MPL-6']:
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

        spaxel = SpaxelBase(15, 15, plateifu=plateifu,
                            cube=cube, maps=maps, modelcube=modelcube,
                            template=template, bintype=bintype)

        assert isinstance(spaxel, SpaxelBase)

        if (spaxel.bintype is not None and spaxel.bintype.binned is True and
                (spaxel._maps or spaxel._modelcube)):
            assert isinstance(spaxel, Bin)
        else:
            assert isinstance(spaxel, Spaxel)

        if spaxel._cube:
            assert len(spaxel.cube_quantities) > 0
        else:
            assert len(spaxel.cube_quantities) == 0

        if spaxel._maps:
            assert len(spaxel.maps_quantities) > 0
        else:
            assert len(spaxel.maps_quantities) == 0

        if spaxel._modelcube:
            assert len(spaxel.modelcube_quantities) > 0
        else:
            assert len(spaxel.modelcube_quantities) == 0

        assert spaxel.plateifu == galaxy_spaxel.plateifu
        assert spaxel.mangaid == galaxy_spaxel.mangaid

    def test_dir(self):

        spaxel = SpaxelBase(15, 15, plateifu='8485-1901', cube=True,
                            maps=True, modelcube=True)

        dir_list = dir(spaxel)

        assert 'flux' in dir_list
        assert 'emline_gflux_ha_6564' in dir_list
        assert 'binned_flux' in dir_list

    def test_getattr(self):

        spaxel = SpaxelBase(15, 15, plateifu='8485-1901', cube=True,
                            maps=True, modelcube=True)

        assert spaxel.flux is not None
        assert spaxel.emline_gflux_ha_6564 is not None
        assert spaxel.binned_flux is not None

    def test_no_inputs(self):

        with pytest.raises(MarvinError) as ee:
            SpaxelBase(0, 0, cube=None, maps=None, modelcube=None)

        assert 'no inputs defined' in str(ee)

    def test_files_maps_modelcube(self, galaxy_spaxel):

        if galaxy_spaxel.release == 'MPL-4':
            modelcube_filename = None
        else:
            modelcube_filename = galaxy_spaxel.modelpath

        spaxel = SpaxelBase(15, 15,
                            cube=galaxy_spaxel.cubepath,
                            maps=galaxy_spaxel.mapspath,
                            modelcube=modelcube_filename)

        assert isinstance(spaxel, SpaxelBase)

        assert isinstance(spaxel._cube, Cube)
        assert isinstance(spaxel._maps, Maps)
        assert isinstance(spaxel._modelcube, ModelCube)

    def test_files_modelcube(self, galaxy_spaxel):

        if galaxy_spaxel.release == 'MPL-4':
            modelcube_filename = None
        else:
            modelcube_filename = galaxy_spaxel.modelpath

        spaxel = SpaxelBase(15, 15,
                            cube=False,
                            maps=False,
                            modelcube=modelcube_filename)

        assert isinstance(spaxel, SpaxelBase)

        assert not isinstance(spaxel._cube, Cube)
        assert not isinstance(spaxel._maps, Maps)
        assert isinstance(spaxel._modelcube, ModelCube)

    def test_files_maps(self, galaxy_spaxel):

        spaxel = SpaxelBase(15, 15,
                            cube=False,
                            maps=galaxy_spaxel.mapspath,
                            modelcube=False)

        assert isinstance(spaxel, SpaxelBase)

        assert not isinstance(spaxel._cube, Cube)
        assert isinstance(spaxel._maps, Maps)
        assert not isinstance(spaxel._modelcube, ModelCube)


class TestBin(object):

    def test_bad_binid(self):

        with pytest.raises(MarvinError) as ee:
            SpaxelBase(0, 0, plateifu='8485-1901', cube=True,
                       maps=True, modelcube=True, bintype='HYB10')

        assert 'not correspond to a valid binid' in str(ee)

    def test_load_all(self):

        bb = SpaxelBase(15, 15, plateifu='8485-1901', cube=True,
                        maps=True, modelcube=True, bintype='HYB10', release='MPL-6')

        assert isinstance(bb, Bin)

        assert len(bb.spaxels) > 0
        assert bb.spaxels[0].loaded is False

        bb.load_all()

        for sp in bb.spaxels:
            assert sp.loaded is True


class TestPickling(object):

    def test_pickling_db_fails(self, temp_scratch, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        spaxel = cube.getSpaxel(1, 3)

        file = temp_scratch.join('test_spaxel.mpf')

        with pytest.raises(MarvinError) as cm:
            spaxel.save(str(file), overwrite=True)

        assert 'objects with data_origin=\'db\' cannot be saved.' in str(cm.value)

    def test_pickling_only_cube_file(self, temp_scratch, galaxy):
        if galaxy.bintype.name != 'SPX':
            pytest.skip("Can't instantiate a Spaxel from a binned Maps.")

        cube = Cube(filename=galaxy.cubepath)
        maps = Maps(filename=galaxy.mapspath)

        spaxel = cube.getSpaxel(1, 3, properties=maps, models=False)

        file = temp_scratch.join('test_spaxel.mpf')

        path_saved = spaxel.save(str(file), overwrite=True)
        assert file.check() is True
        assert os.path.exists(path_saved)

        del spaxel

        spaxel_restored = SpaxelBase.restore(str(file))
        assert spaxel_restored is not None
        assert isinstance(spaxel_restored, SpaxelBase)

        assert spaxel_restored._cube is not None
        assert spaxel_restored._cube.data_origin == 'file'
        assert isinstance(spaxel_restored._cube.data, astropy.io.fits.HDUList)

        assert spaxel_restored._maps is not None
        assert spaxel_restored._maps.data_origin == 'file'
        assert isinstance(spaxel_restored._maps.data, astropy.io.fits.HDUList)

    @pytest.mark.parametrize('mpl', ['MPL-5'])
    def test_pickling_all_api(self, monkeypatch, temp_scratch, galaxy, mpl):
        monkeypatch.setattr(config, 'release', mpl)
        drpver, __ = config.lookUpVersions()

        cube = Cube(plateifu=galaxy.plateifu, mode='remote')
        maps = Maps(plateifu=galaxy.plateifu, mode='remote')
        modelcube = ModelCube(plateifu=galaxy.plateifu, mode='remote')
        spaxel = cube.getSpaxel(1, 3, properties=maps, models=modelcube)

        assert spaxel._cube.data_origin == 'api'
        assert spaxel._maps.data_origin == 'api'
        assert spaxel._modelcube.data_origin == 'api'

        file = temp_scratch.join('test_spaxel_api.mpf')

        path_saved = spaxel.save(str(file), overwrite=True)
        assert file.check() is True
        assert os.path.exists(path_saved)

        del spaxel

        spaxel_restored = SpaxelBase.restore(str(file))
        assert spaxel_restored is not None
        assert isinstance(spaxel_restored, SpaxelBase)

        assert spaxel_restored._cube is not None
        assert isinstance(spaxel_restored._cube, Cube)
        assert spaxel_restored._cube.data_origin == 'api'
        assert spaxel_restored._cube.data is None
        assert spaxel_restored._cube.header['VERSDRP3'] == drpver

        assert spaxel_restored._maps is not None
        assert isinstance(spaxel_restored._maps, Maps)
        assert spaxel_restored._maps.data_origin == 'api'
        assert spaxel_restored._maps.data is None

        assert spaxel_restored._modelcube is not None
        assert isinstance(spaxel_restored._modelcube, ModelCube)
        assert spaxel_restored._modelcube.data_origin == 'api'
        assert spaxel_restored._modelcube.data is None
