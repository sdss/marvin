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

from marvin.tests import marvin_test_if_class

from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube
from marvin.tools.quantities import Spectrum
from marvin.tools.spaxel import SpaxelBase, Spaxel, Bin
from marvin.tests import marvin_test_if


spaxel_modes = [True, False, 'object']


def _get_spaxel_helper(object, x, y, **kwargs):
    try:
        spaxel = object.getSpaxel(x=x, y=y, **kwargs)
        return spaxel
    except MarvinError as ee:
        assert 'do not correspond to a valid binid' in str(ee)
        pytest.skip()


@pytest.fixture(params=itertools.product(spaxel_modes, spaxel_modes, spaxel_modes))
def cube_maps_modelcube_modes(request):
    return request.param


@marvin_test_if_class(mark='include', galaxy=dict(plateifu=['8485-1901']))
class TestSpaxel(object):

    def test_SpaxelBase(self, galaxy, cube_maps_modelcube_modes):

        plateifu = galaxy.plateifu
        bintype = galaxy.bintype.name
        template = galaxy.template.name
        release = galaxy.release
        x = galaxy.dap['x']
        y = galaxy.dap['y']

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

        spaxel = SpaxelBase(x, y, plateifu=plateifu,
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

        assert spaxel.plateifu == galaxy.plateifu
        assert spaxel.mangaid == galaxy.mangaid

        assert isinstance(spaxel.getCube(), Cube)
        assert isinstance(spaxel.getMaps(), Maps)

        if release != 'MPL-4':
            assert isinstance(spaxel.getModelCube(), ModelCube)

    def test_dir(self, galaxy):

        x = galaxy.dap['x']
        y = galaxy.dap['y']

        spaxel = SpaxelBase(x, y, plateifu='8485-1901', cube=True,
                            maps=True, modelcube=True)

        dir_list = dir(spaxel)

        assert 'flux' in dir_list
        assert 'emline_gflux_ha_6564' in dir_list
        assert 'binned_flux' in dir_list

    def test_getattr(self, galaxy):

        x = galaxy.dap['x']
        y = galaxy.dap['y']

        spaxel = SpaxelBase(x, y, plateifu='8485-1901', cube=True,
                            maps=True, modelcube=True)

        assert spaxel.flux is not None
        assert spaxel.emline_gflux_ha_6564 is not None
        assert spaxel.binned_flux is not None

    def test_no_inputs(self):

        with pytest.raises(MarvinError) as ee:
            SpaxelBase(0, 0, cube=None, maps=None, modelcube=None)

        assert 'no inputs defined' in str(ee)

    def test_files_maps_modelcube(self, galaxy):

        x = galaxy.dap['x']
        y = galaxy.dap['y']

        if galaxy.release == 'MPL-4':
            modelcube_filename = None
        else:
            modelcube_filename = galaxy.modelpath

        spaxel = SpaxelBase(x, y,
                            cube=galaxy.cubepath,
                            maps=galaxy.mapspath,
                            modelcube=modelcube_filename)

        assert isinstance(spaxel, SpaxelBase)

        assert isinstance(spaxel._cube, Cube)
        assert isinstance(spaxel._maps, Maps)

        if galaxy.release != 'MPL-4':
            assert isinstance(spaxel._modelcube, ModelCube)

    def test_files_modelcube(self, galaxy):

        x = galaxy.dap['x']
        y = galaxy.dap['y']

        if galaxy.release == 'MPL-4':
            pytest.skip()
        else:
            modelcube_filename = galaxy.modelpath

        spaxel = SpaxelBase(x, y,
                            cube=False,
                            maps=False,
                            modelcube=modelcube_filename)

        assert isinstance(spaxel, SpaxelBase)

        assert not isinstance(spaxel._cube, Cube)
        assert not isinstance(spaxel._maps, Maps)

        if galaxy.release != 'MPL-4':
            assert isinstance(spaxel._modelcube, ModelCube)

    def test_files_maps(self, galaxy):

        x = galaxy.dap['x']
        y = galaxy.dap['y']

        spaxel = SpaxelBase(x, y,
                            cube=False,
                            maps=galaxy.mapspath,
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

        assert 'do not correspond to a valid binid' in str(ee) or 'invalid bintype' in str(ee)

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

    def test_pickling_all_api(self, temp_scratch, galaxy):
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


class TestMaskbit(object):

    @marvin_test_if(mark='include', galaxy=dict(release=['MPL-4']))
    def test_quality_flags_mpl4(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        sp = maps.getSpaxel(0, 0, model=True)
        assert len(sp.quality_flags) == 1

    @marvin_test_if(mark='skip', galaxy=dict(release=['MPL-4']))
    def test_quality_flags(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        sp = maps.getSpaxel(0, 0, model=True)
        assert len(sp.quality_flags) == 2

    @pytest.mark.parametrize('flag',
                             ['manga_target1',
                              'manga_target2',
                              'manga_target3',
                              'target_flags'])
    def test_flag(self, flag, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        sp = maps[0, 0]
        assert getattr(sp, flag, None) is not None


class TestCubeGetSpaxel(object):

    def _dropNones(self, **kwargs):
        for k, v in list(kwargs.items()):
            if v is None:
                del kwargs[k]
        return kwargs

    @pytest.mark.parametrize(
        'x, y, ra, dec, excType, message',
        [(1, None, 1, None, AssertionError, 'Either use (x, y) or (ra, dec)'),
         (1, None, 1, 1, AssertionError, 'Either use (x, y) or (ra, dec)'),
         (1, None, None, None, AssertionError, 'Specify both x and y'),
         (None, 1, None, None, AssertionError, 'Specify both x and y'),
         (None, None, 1, None, AssertionError, 'Specify both ra and dec'),
         (None, None, None, 1, AssertionError, 'Specify both ra and dec'),
         (None, None, None, None, ValueError, 'You need to specify either (x, y) or (ra, dec)'),
         (-50, 1, None, None, MarvinError, 'some indices are out of limits'),
         (50, 1, None, None, MarvinError, 'some indices are out of limits'),
         (1, -50, None, None, MarvinError, 'some indices are out of limits'),
         (1, 50, None, None, MarvinError, 'some indices are out of limits'),
         (None, None, 1., 1., MarvinError, 'some indices are out of limits'),
         (None, None, 100, 60, MarvinError, 'some indices are out of limits'),
         (None, None, 232.546383, 1., MarvinError, 'some indices are out of limits'),
         (None, None, 1., 48.6883954, MarvinError, 'some indices are out of limits')],
        ids=['x-ra', 'x-ra-dec', 'x', 'y', 'ra', 'dec', 'no-inputs', '-50-1', '50-1', '1--50',
             '1-50', '1-1', '100-60', '232.5-1', '1-48.6'])
    def test_getSpaxel_inputs(self, galaxy, x, y, ra, dec, excType, message):
        """Tests exceptions when getSpaxel gets inappropriate inputs."""
        kwargs = self._dropNones(x=x, y=y, ra=ra, dec=dec)

        with pytest.raises(excType) as ee:
            cube = Cube(plateifu=galaxy.plateifu, release=galaxy.release)
            cube.getSpaxel(**kwargs)

        assert message in str(ee.value)

    @pytest.mark.parametrize('coord, xyorig',
                             [('xy', 'lower'),
                              ('xy', 'center'),
                              ('radec', None)])
    def test_getSpaxel_flux(self, cube, galaxy, coord, xyorig):
        if coord == 'xy':
            x = galaxy.spaxel['x'] if xyorig == 'lower' else galaxy.spaxel['x_cen']
            y = galaxy.spaxel['y'] if xyorig == 'lower' else galaxy.spaxel['y_cen']
            params = {'x': x, 'y': y, 'xyorig': xyorig}
        elif coord == 'radec':
            ra = galaxy.spaxel['ra']
            dec = galaxy.spaxel['dec']
            params = {'ra': ra, 'dec': dec}

        spaxel = cube.getSpaxel(**params)
        flux = spaxel.flux.value
        assert pytest.approx(flux[galaxy.spaxel['specidx']], galaxy.spaxel['flux'])

    @pytest.mark.parametrize('monkeyconfig',
                             [('sasurl', 'http://www.averywrongurl.com')],
                             ids=['wrongurl'], indirect=True)
    def test_getSpaxel_remote_fail_badresponse(self, monkeyconfig):
        assert config.urlmap is not None

        with pytest.raises(MarvinError) as cm:
            Cube(mangaid='1-209232', mode='remote')

        assert 'Failed to establish a new connection' in str(cm.value)

    @pytest.mark.parametrize('monkeyconfig',
                             [('release', 'MPL-5')],
                             ids=['mpl5'], indirect=True)
    def test_getSpaxel_remote_drpver_differ_from_global(self, galaxy, monkeyconfig):
        if galaxy.release == 'MPL-5':
            pytest.skip('Skipping release for forced global MPL-5')

        assert config.release == 'MPL-5'

        cube = Cube(plateifu=galaxy.plateifu, mode='remote', release=galaxy.release)
        expected = galaxy.spaxel['flux']

        spectrum = cube.getSpaxel(ra=galaxy.spaxel['ra'], dec=galaxy.spaxel['dec']).flux
        assert pytest.approx(spectrum.value[galaxy.spaxel['specidx']], expected)

    def test_getspaxel_matches_file_db_remote(self, galaxy):

        cube_file = Cube(filename=galaxy.cubepath)
        cube_db = Cube(plateifu=galaxy.plateifu)
        cube_api = Cube(plateifu=galaxy.plateifu, mode='remote')

        assert cube_file.data_origin == 'file'
        assert cube_db.data_origin == 'db'
        assert cube_api.data_origin == 'api'

        xx = galaxy.spaxel['x']
        yy = galaxy.spaxel['y']
        spec_idx = galaxy.spaxel['specidx']
        flux = galaxy.spaxel['flux']
        ivar = galaxy.spaxel['ivar']
        mask = galaxy.spaxel['mask']

        spaxel_slice_file = cube_file[yy, xx]
        spaxel_slice_db = cube_db[yy, xx]
        spaxel_slice_api = cube_api[yy, xx]

        assert pytest.approx(spaxel_slice_file.flux.value[spec_idx], flux)
        assert pytest.approx(spaxel_slice_db.flux.value[spec_idx], flux)
        assert pytest.approx(spaxel_slice_api.flux.value[spec_idx], flux)

        assert pytest.approx(spaxel_slice_file.flux.ivar[spec_idx], ivar)
        assert pytest.approx(spaxel_slice_db.flux.ivar[spec_idx], ivar)
        assert pytest.approx(spaxel_slice_api.flux.ivar[spec_idx], ivar)

        assert pytest.approx(spaxel_slice_file.flux.mask[spec_idx], mask)
        assert pytest.approx(spaxel_slice_db.flux.mask[spec_idx], mask)
        assert pytest.approx(spaxel_slice_api.flux.mask[spec_idx], mask)

        xx_cen = galaxy.spaxel['x_cen']
        yy_cen = galaxy.spaxel['y_cen']

        try:
            spaxel_getspaxel_file = cube_file.getSpaxel(x=xx_cen, y=yy_cen)
            spaxel_getspaxel_db = cube_db.getSpaxel(x=xx_cen, y=yy_cen)
            spaxel_getspaxel_api = cube_api.getSpaxel(x=xx_cen, y=yy_cen)
        except MarvinError as ee:
            assert 'do not correspond to a valid binid' in str(ee)
            pytest.skip()

        assert pytest.approx(spaxel_getspaxel_file.flux.value[spec_idx], flux)
        assert pytest.approx(spaxel_getspaxel_db.flux.value[spec_idx], flux)
        assert pytest.approx(spaxel_getspaxel_api.flux.value[spec_idx], flux)

        assert pytest.approx(spaxel_getspaxel_file.flux.ivar[spec_idx], ivar)
        assert pytest.approx(spaxel_getspaxel_db.flux.ivar[spec_idx], ivar)
        assert pytest.approx(spaxel_getspaxel_api.flux.ivar[spec_idx], ivar)

        assert pytest.approx(spaxel_getspaxel_file.flux.mask[spec_idx], mask)
        assert pytest.approx(spaxel_getspaxel_db.flux.mask[spec_idx], mask)
        assert pytest.approx(spaxel_getspaxel_api.flux.mask[spec_idx], mask)


class TestMapsGetSpaxel(object):

    def _get_maps_kwargs(self, galaxy, data_origin):

        if data_origin == 'file':
            maps_kwargs = dict(filename=galaxy.mapspath)
        else:
            maps_kwargs = dict(plateifu=galaxy.plateifu, release=galaxy.release,
                               bintype=galaxy.bintype, template=galaxy.template,
                               mode='local' if data_origin == 'db' else 'remote')

        return maps_kwargs

    def test_get_spaxel(self, galaxy, data_origin):

        maps = Maps(**self._get_maps_kwargs(galaxy, data_origin))

        spaxel = _get_spaxel_helper(maps, 15, 8, xyorig='lower')

        if maps.is_binned():
            assert isinstance(spaxel, Bin)
        else:
            assert isinstance(spaxel, Spaxel)
            expected = galaxy.stellar_vel_ivar_x15_y8_lower[galaxy.release][galaxy.template.name]
            assert spaxel.maps_quantities['stellar_vel'].ivar == pytest.approx(expected, abs=1e-6)

        assert len(spaxel.maps_quantities.keys()) > 0

    def test_get_spaxel_test2(self, galaxy, data_origin):

        maps = Maps(**self._get_maps_kwargs(galaxy, data_origin))

        spaxel = _get_spaxel_helper(maps, 5, 5)

        if maps.is_binned():
            assert isinstance(spaxel, Bin)
        else:
            assert isinstance(spaxel, Spaxel)

        assert len(spaxel.maps_quantities.keys()) > 0

    def test_get_spaxel_no_db(self, galaxy, exporigin):
        """Tests getting an spaxel if there is no DB."""

        maps = Maps(**self._get_maps_kwargs(galaxy, exporigin))
        spaxel = _get_spaxel_helper(maps, 5, 5)

        assert spaxel.getMaps().data_origin == exporigin

        if maps.is_binned():
            assert isinstance(spaxel, Bin)
        else:
            assert isinstance(spaxel, Spaxel)

        assert len(spaxel.maps_quantities.keys()) > 0


@marvin_test_if_class(mark='skip', galaxy=dict(release=['MPL-4']))
class TestModelCubeGetSpaxel(object):

    def _test_getspaxel(self, spaxel, galaxy):

        spaxel_drpver, spaxel_dapver = config.lookUpVersions(spaxel.release)

        assert spaxel_drpver == galaxy.drpver
        assert spaxel_dapver == galaxy.dapver
        assert spaxel.plateifu == galaxy.plateifu
        assert spaxel.mangaid == galaxy.mangaid

        assert spaxel.getModelCube() is not None
        assert spaxel.getModelCube().bintype == galaxy.bintype
        assert spaxel.getModelCube().template == galaxy.template

        assert spaxel.template == galaxy.template
        assert spaxel.template == galaxy.template

        assert spaxel._parent_shape == tuple(galaxy.shape)

        assert isinstance(spaxel.binned_flux, Spectrum)
        assert isinstance(spaxel.full_fit, Spectrum)
        assert isinstance(spaxel.emline_fit, Spectrum)

    def test_getspaxel(self, galaxy, data_origin):

        if data_origin == 'file':
            kwargs = {'filename': galaxy.modelpath}
        elif data_origin == 'db':
            kwargs = {'plateifu': galaxy.plateifu}
        elif data_origin == 'api':
            kwargs = {'plateifu': galaxy.plateifu, 'mode': 'remote'}

        model_cube = ModelCube(bintype=galaxy.bintype, template=galaxy.template,
                               release=galaxy.release, **kwargs)
        spaxel = _get_spaxel_helper(model_cube, 1, 2)
        self._test_getspaxel(spaxel, galaxy)

    def test_getspaxel_db_api_model(self, galaxy):

        model_cube = ModelCube(plateifu=galaxy.plateifu,
                               bintype=galaxy.bintype, template=galaxy.template,
                               release=galaxy.release, )
        spaxel = _get_spaxel_helper(model_cube, 1, 2, properties=False, drp=False)
        self._test_getspaxel(spaxel, galaxy)
        assert isinstance(spaxel.getCube(), Cube)
        assert 'flux' not in spaxel.cube_quantities
        assert isinstance(spaxel.getMaps(), Maps)
        assert len(spaxel.maps_quantities) == 0

    def test_getspaxel_matches_file_db_remote(self, galaxy):

        modelcube_file = ModelCube(filename=galaxy.modelpath,
                                   bintype=galaxy.bintype, template=galaxy.template,
                                   release=galaxy.release)
        modelcube_db = ModelCube(mangaid=galaxy.mangaid, bintype=galaxy.bintype,
                                 template=galaxy.template, release=galaxy.release)
        modelcube_api = ModelCube(mangaid=galaxy.mangaid, mode='remote',
                                  bintype=galaxy.bintype, template=galaxy.template,
                                  release=galaxy.release)

        assert modelcube_file.data_origin == 'file'
        assert modelcube_db.data_origin == 'db'
        assert modelcube_api.data_origin == 'api'

        idx = galaxy.spaxel['specidx']
        flux = galaxy.spaxel['model_flux']
        ivar = galaxy.spaxel['model_ivar']
        mask = galaxy.spaxel['model_mask']

        xx_cen = galaxy.spaxel['x_cen']
        yy_cen = galaxy.spaxel['y_cen']

        try:
            spaxel_getspaxel_file = modelcube_file.getSpaxel(x=xx_cen, y=yy_cen)
            spaxel_getspaxel_db = modelcube_db.getSpaxel(x=xx_cen, y=yy_cen)
            spaxel_getspaxel_api = modelcube_api.getSpaxel(x=xx_cen, y=yy_cen)
        except MarvinError as ee:
            assert 'do not correspond to a valid binid' in str(ee)
            pytest.skip()

        assert pytest.approx(spaxel_getspaxel_file.binned_flux[idx], flux)
        assert pytest.approx(spaxel_getspaxel_db.binned_flux[idx], flux)
        assert pytest.approx(spaxel_getspaxel_api.binned_flux[idx], flux)

        assert pytest.approx(spaxel_getspaxel_file.binned_flux.ivar[idx], ivar)
        assert pytest.approx(spaxel_getspaxel_db.binned_flux.ivar[idx], ivar)
        assert pytest.approx(spaxel_getspaxel_api.binned_flux.ivar[idx], ivar)

        assert pytest.approx(spaxel_getspaxel_file.binned_flux.mask[idx], mask)
        assert pytest.approx(spaxel_getspaxel_db.binned_flux.mask[idx], mask)
        assert pytest.approx(spaxel_getspaxel_api.binned_flux.mask[idx], mask)
