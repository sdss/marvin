#!/usr/bin/env python
# encoding: utf-8
#
# test_spaxels.py
#
# Created by José Sánchez-Gallego on Sep 9, 2016.


from __future__ import division, print_function, absolute_import

import os

import pytest
import astropy.io.fits

from marvin import config
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube
from marvin.core.exceptions import MarvinError
from marvin.tools.analysis_props import DictOfProperties
from marvin.tools.spaxel import Spaxel
from marvin.tools.spectrum import Spectrum


class TestSpaxelInit(object):

    def _spaxel_init(self, spaxel, cube, maps, spectrum):

        args = {'cube': cube, 'maps': maps, 'spectrum': spectrum}
        objs = {'cube': Cube, 'maps': Maps, 'spectrum': Spectrum}

        for key in objs:
            if args[key]:
                assert isinstance(getattr(spaxel, key), objs[key])
            else:
                assert getattr(spaxel, key) is None

        if maps:
            assert len(spaxel.properties) > 0
            assert isinstance(spaxel.properties, DictOfProperties)
        else:
            assert len(spaxel.properties) == 0

    def test_no_cube_no_maps_db(self, galaxy):
        spaxel = Spaxel(x=15, y=16, plateifu=galaxy.plateifu)
        self._spaxel_init(spaxel, cube=True, maps=True, spectrum=True)

    def test_cube_false_no_maps_db(self, galaxy):
        spaxel = Spaxel(x=15, y=16, plateifu=galaxy.plateifu, cube=False)
        self._spaxel_init(spaxel, cube=False, maps=True, spectrum=False)

    def test_no_cube_maps_false_db(self, galaxy):
        spaxel = Spaxel(x=15, y=16, plateifu=galaxy.plateifu, maps=False)
        self._spaxel_init(spaxel, cube=True, maps=False, spectrum=True)

    def test_cube_object_db(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube)

        self._spaxel_init(spaxel, cube=True, maps=True, spectrum=True)

    def test_cube_object_maps_false_db(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=False)

        self._spaxel_init(spaxel, cube=True, maps=False, spectrum=True)

    def test_cube_maps_object_filename(self, galaxy):
        if galaxy.bintype not in ['SPX', 'NONE']:
            pytest.skip("Can't instantiate a Spaxel from a binned Maps.")

        cube = Cube(filename=galaxy.cubepath)
        maps = Maps(filename=galaxy.mapspath, bintype=galaxy.bintype, release=galaxy.release)
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=maps)

        assert cube._drpver == galaxy.drpver
        assert spaxel._drpver == galaxy.drpver
        assert maps._drpver == galaxy.drpver
        assert maps._dapver == galaxy.dapver
        assert spaxel._dapver == galaxy.dapver

        self._spaxel_init(spaxel, cube=True, maps=True, spectrum=True)

    def test_cube_object_api(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu, mode='remote')
        spaxel = Spaxel(x=15, y=16, cube=cube)

        self._spaxel_init(spaxel, cube=True, maps=True, spectrum=True)

    def test_cube_maps_object_api(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu, mode='remote')
        maps = Maps(plateifu=galaxy.plateifu, mode='remote')
        spaxel = Spaxel(x=15, y=16, cube=cube, maps=maps)

        self._spaxel_init(spaxel, cube=True, maps=True, spectrum=True)

    def test_db_maps_template(self, galaxy):
        spaxel = Spaxel(x=15, y=16, cube=False, modelcube=False, maps=True,
                        plateifu=galaxy.plateifu, template_kin=galaxy.template)
        assert spaxel.maps.template_kin == galaxy.template

    def test_api_maps_invalid_template(self, galaxy):
        with pytest.raises(AssertionError) as cm:
            Spaxel(x=15, y=16, cube=False, modelcube=False, maps=True, plateifu=galaxy.plateifu,
                   template_kin='invalid-template')
        assert 'invalid template_kin' in str(cm.value)

    def test_load_false(self, galaxy):
        spaxel = Spaxel(plateifu=galaxy.plateifu, x=15, y=16, load=False)

        assert not spaxel.loaded
        assert spaxel.cube
        assert spaxel.maps
        assert spaxel.modelcube
        assert len(spaxel.properties) == 0
        assert spaxel.spectrum is None

        spaxel.load()

        self._spaxel_init(spaxel, cube=True, maps=True, spectrum=True)

    def test_fails_unbinned_maps(self, galaxy):
        if galaxy.bintype in ['SPX', 'NONE']:
            pytest.skip("Can instantiate a Spaxel from a binned Maps.")

        maps = Maps(plateifu=galaxy.plateifu, bintype=galaxy.bintype, release=galaxy.release)

        with pytest.raises(MarvinError) as cm:
            Spaxel(x=15, y=16, plateifu=galaxy.plateifu, maps=maps)

        assert 'cannot instantiate a Spaxel from a binned Maps.' in str(cm.value)

    def test_spaxel_ra_dec(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube)

        assert pytest.approx(spaxel.ra, 232.54512)
        assert pytest.approx(spaxel.dec, 48.690062)

    @pytest.mark.parametrize('mpl', ['MPL-4', 'MPL-5'])
    def test_release(self, monkeypatch, galaxy, mpl):
        monkeypatch.setattr(config, 'release', mpl)

        cube = Cube(plateifu=galaxy.plateifu)
        spaxel = Spaxel(x=15, y=16, cube=cube)

        assert spaxel.release == mpl

        with pytest.raises(MarvinError) as cm:
            spaxel.release = 'a'
            assert 'the release cannot be changed' in str(cm.value)


class TestPickling(object):

    def test_pickling_db_fails(self, temp_scratch, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        spaxel = cube.getSpaxel(x=1, y=3)

        file = temp_scratch.join('test_spaxel.mpf')

        with pytest.raises(MarvinError) as cm:
            spaxel.save(str(file), overwrite=True)

        assert 'objects with data_origin=\'db\' cannot be saved.' in str(cm.value)

    def test_pickling_only_cube_file(self, temp_scratch, galaxy):
        if galaxy.bintype != 'SPX':
            pytest.skip("Can't instantiate a Spaxel from a binned Maps.")

        cube = Cube(filename=galaxy.cubepath)
        maps = Maps(filename=galaxy.mapspath)

        spaxel = cube.getSpaxel(x=1, y=3, properties=maps, modelcube=False)

        file = temp_scratch.join('test_spaxel.mpf')

        path_saved = spaxel.save(str(file), overwrite=True)
        assert file.check() is True
        assert os.path.exists(path_saved)

        del spaxel

        spaxel_restored = Spaxel.restore(str(file))
        assert spaxel_restored is not None
        assert isinstance(spaxel_restored, Spaxel)

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
        spaxel = cube.getSpaxel(x=1, y=3, properties=maps, modelcube=modelcube)

        assert spaxel.cube.data_origin == 'api'
        assert spaxel.maps.data_origin == 'api'
        assert spaxel.modelcube.data_origin == 'api'

        file = temp_scratch.join('test_spaxel_api.mpf')

        path_saved = spaxel.save(str(file), overwrite=True)
        assert file.check() is True
        assert os.path.exists(path_saved)

        del spaxel

        spaxel_restored = Spaxel.restore(str(file))
        assert spaxel_restored is not None
        assert isinstance(spaxel_restored, Spaxel)

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
