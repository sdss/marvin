#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Nov 1, 2017
# @Filename: test_spaxel.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import absolute_import, division, print_function

import itertools
import os

import astropy.io.fits
import pytest

from marvin import config
from marvin.core.exceptions import MarvinDeprecationError, MarvinError
from tests import marvin_test_if, marvin_test_if_class
from tests.conftest import set_the_config
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube
from marvin.tools.quantities import Spectrum
from marvin.tools.spaxel import Spaxel
from marvin.utils.datamodel.dap import Property
from marvin.web.controllers.galaxy import get_flagged_regions


pytestmark = pytest.mark.usefixtures("checkdb")


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

        spaxel = Spaxel(x, y, plateifu=plateifu,
                        cube=cube, maps=maps, modelcube=modelcube,
                        template=template, bintype=bintype)

        assert isinstance(spaxel, Spaxel)

        if (spaxel.bintype is not None and spaxel.bintype.binned is True and
                (spaxel._maps or spaxel._modelcube)):
            assert isinstance(spaxel, Spaxel)
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

        spaxel = Spaxel(x, y, plateifu='8485-1901', cube=True,
                        maps=True, modelcube=True)

        dir_list = dir(spaxel)

        assert 'flux' in dir_list
        assert 'emline_gflux_ha_6564' in dir_list
        assert 'binned_flux' in dir_list

    def test_getattr(self, galaxy):

        x = galaxy.dap['x']
        y = galaxy.dap['y']

        spaxel = Spaxel(x, y, plateifu='8485-1901', cube=True,
                        maps=True, modelcube=True)

        assert spaxel.flux is not None
        assert spaxel.emline_gflux_ha_6564 is not None
        assert spaxel.binned_flux is not None

    @pytest.mark.parametrize('force',
                             [('cube'),
                              ('maps'),
                              ('modelcube')],
                             ids=[])
    def test_force_load(self, galaxy, force):

        x = galaxy.dap['x']
        y = galaxy.dap['y']
        spaxel = Spaxel(x, y, plateifu=galaxy.plateifu, cube=True,
                        maps=False, modelcube=False)

        assert spaxel.cube_quantities is not None
        assert spaxel.maps_quantities == {}
        assert spaxel.modelcube_quantities == {}

        spaxel.load(force=force)

        if force == 'cube':
            assert spaxel.cube_quantities is not None
        elif force == 'maps':
            assert spaxel.maps_quantities is not None
        elif force == 'modelcube':
            assert spaxel.modelcube_quantities is not None

    def test_wrong_force_load(self, galaxy):

        x = galaxy.dap['x']
        y = galaxy.dap['y']
        spaxel = Spaxel(x, y, plateifu=galaxy.plateifu, cube=True,
                        maps=False, modelcube=False)

        with pytest.raises(AssertionError) as ee:
            spaxel.load(force='crap')

        assert 'force can only be cube, maps, or modelcube' in str(ee)

    def test_no_inputs(self):

        with pytest.raises(MarvinError) as ee:
            Spaxel(0, 0, cube=None, maps=None, modelcube=None)

        assert 'no inputs defined' in str(ee)

    def test_files_maps_modelcube(self, galaxy):

        x = galaxy.dap['x']
        y = galaxy.dap['y']

        if galaxy.release == 'MPL-4':
            modelcube_filename = None
        else:
            modelcube_filename = galaxy.modelpath

        spaxel = Spaxel(x, y,
                        cube=galaxy.cubepath,
                        maps=galaxy.mapspath,
                        modelcube=modelcube_filename)

        assert isinstance(spaxel, Spaxel)

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

        spaxel = Spaxel(x, y,
                        cube=False,
                        maps=False,
                        modelcube=modelcube_filename)

        assert isinstance(spaxel, Spaxel)

        assert not isinstance(spaxel._cube, Cube)
        assert not isinstance(spaxel._maps, Maps)

        if galaxy.release != 'MPL-4':
            assert isinstance(spaxel._modelcube, ModelCube)

    def test_files_maps(self, galaxy):

        x = galaxy.dap['x']
        y = galaxy.dap['y']

        spaxel = Spaxel(x, y,
                        cube=False,
                        maps=galaxy.mapspath,
                        modelcube=False)

        assert isinstance(spaxel, Spaxel)

        assert not isinstance(spaxel._cube, Cube)
        assert isinstance(spaxel._maps, Maps)
        assert not isinstance(spaxel._modelcube, ModelCube)


#@pytest.mark.usefixtures('db_off')
class TestBinInfo(object):

    def test_bad_binid(self):

        spaxel = Spaxel(0, 0, plateifu='8485-1901', cube=True,
                        maps=True, modelcube=True, bintype='HYB10')

        with pytest.raises(MarvinError) as ee:
            spaxel.stellar_vel.bin.get_bin_spaxels()

        assert 'do not correspond to a valid binid' in str(ee)

    def test_load_all(self):

        set_the_config('DR17')
        spaxel = Spaxel(26, 13, plateifu='8485-1901', cube=True,
                        maps=True, modelcube=True, bintype='HYB10', release='DR17')

        assert isinstance(spaxel, Spaxel)

        bin_spaxels = spaxel.stellar_vel.bin.get_bin_spaxels(lazy=False)

        assert len(bin_spaxels) > 0
        assert bin_spaxels[0].loaded is True

    def test_correct_binid(self):
        """Checks if the binid of the bin spaxels is the correct one (#457)"""

        maps = Maps(plateifu='8485-1901', release='DR17', bintype='HYB10')
        spaxel = maps[22, 14]

        assert isinstance(spaxel, Spaxel)
        assert spaxel.x == 14, spaxel.y == 22

        bin_spaxels = spaxel.stellar_vel.bin.get_bin_spaxels()

        for sp in bin_spaxels:

            sp.load()
            assert sp.stellar_vel.bin.binid == spaxel.stellar_vel.bin.binid

            sp_bin = maps[sp.y, sp.x]
            assert sp_bin.stellar_vel.bin.binid == spaxel.stellar_vel.bin.binid

    def test_hasbin(self):
        maps = Maps(plateifu='8485-1901', release='DR17', bintype='HYB10')
        spaxel = maps[22, 14]
        sv = spaxel.stellar_vel
        assert sv.bin is not None

    def test_hasmap(self):
        maps = Maps(plateifu='8485-1901', release='DR17', bintype='HYB10')
        spaxel = maps[22, 14]
        b = spaxel.stellar_vel.bin
        assert isinstance(b._parent, Maps)
        assert isinstance(b._datamodel, Property)
        assert b._datamodel.name == 'stellar_vel'
        assert b.binid_map is not None


class TestPickling(object):

    @pytest.mark.uses_db
    def test_pickling_db_fails(self, temp_scratch, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        spaxel = cube.getSpaxel(1, 3)

        file = temp_scratch / 'test_spaxel.mpf'

        with pytest.raises(MarvinError, match="objects with data_origin='db' cannot be saved."):
            spaxel.save(str(file), overwrite=True)

        #assert 'objects with data_origin=\'db\' cannot be saved.' in str(cm.value)

    def test_pickling_only_cube_file(self, temp_scratch, galaxy):
        if galaxy.bintype.name != 'SPX':
            pytest.skip("Can't instantiate a Spaxel from a binned Maps.")

        cube = Cube(filename=galaxy.cubepath)
        maps = Maps(filename=galaxy.mapspath)

        spaxel = cube.getSpaxel(1, 3, maps=maps, modelcube=False)

        file = temp_scratch / 'test_spaxel.mpf'

        path_saved = spaxel.save(str(file), overwrite=True)
        assert file.exists() is True
        assert os.path.exists(path_saved)

        del spaxel

        spaxel_restored = Spaxel.restore(str(file))
        assert spaxel_restored is not None
        assert isinstance(spaxel_restored, Spaxel)

        assert spaxel_restored._cube is not None
        assert spaxel_restored._cube.data_origin == 'file'
        assert isinstance(spaxel_restored._cube.data, astropy.io.fits.HDUList)

        assert spaxel_restored._maps is not None
        assert spaxel_restored._maps.data_origin == 'file'
        assert isinstance(spaxel_restored._maps.data, astropy.io.fits.HDUList)

    @pytest.mark.uses_web
    def test_pickling_all_api(self, temp_scratch, galaxy):
        drpver, __ = config.lookUpVersions()

        cube = Cube(plateifu=galaxy.plateifu, mode='remote')
        maps = Maps(plateifu=galaxy.plateifu, mode='remote')
        modelcube = ModelCube(plateifu=galaxy.plateifu, mode='remote')
        spaxel = cube.getSpaxel(1, 3, maps=maps, modelcube=modelcube)

        assert spaxel._cube.data_origin == 'api'
        assert spaxel._maps.data_origin == 'api'
        assert spaxel._modelcube.data_origin == 'api'

        file = temp_scratch / 'test_spaxel_api.mpf'

        path_saved = spaxel.save(str(file), overwrite=True)
        assert file.exists() is True
        assert os.path.exists(path_saved)

        del spaxel

        spaxel_restored = Spaxel.restore(str(file))
        assert spaxel_restored is not None
        assert isinstance(spaxel_restored, Spaxel)

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

    def test_pickling_data(self, temp_scratch, galaxy):

        drpver, __ = config.lookUpVersions()

        maps = Maps(filename=galaxy.mapspath)
        modelcube = ModelCube(filename=galaxy.modelpath)
        spaxel = maps.getSpaxel(25, 15, xyorig='lower', cube=False, modelcube=modelcube)

        file = temp_scratch / 'test_spaxel.mpf'

        path_saved = spaxel.save(str(file), overwrite=True)
        assert file.exists() is True
        assert os.path.exists(path_saved)

        del spaxel

        spaxel_restored = Spaxel.restore(str(file))

        assert spaxel_restored.stellar_vel.value is not None
        assert spaxel_restored.stellar_vel.bin.binid is not None


class TestMaskbit(object):

    # @marvin_test_if(mark='include', galaxy=dict(release=['DR17']))
    # def test_quality_flags_mpl4(self, galaxy):
    #     maps = Maps(plateifu=galaxy.plateifu)
    #     sp = maps.getSpaxel(0, 0, modelcube=True)
    #     assert len(sp.quality_flags) == 1

    #@marvin_test_if(mark='skip', galaxy=dict(release=['DR17']))
    def test_quality_flags(self, galaxy):
        maps = Maps(filename=galaxy.mapspath)
        sp = maps.getSpaxel(0, 0, modelcube=True)
        assert len(sp.quality_flags) == 2

    def test_flagged_regions(self, cube, galaxy):
        params = {'x': galaxy.spaxel['x'], 'y': galaxy.spaxel['y'], 'xyorig': 'lower'}
        spaxel = cube.getSpaxel(**params)
        donotuse = spaxel.flux.pixmask.get_mask(['DONOTUSE'])
        val = 1024
        assert val in donotuse
        badspots = get_flagged_regions(donotuse, value=val)
        assert badspots == galaxy.badspots


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
        abs = 1.e-7
        assert flux[galaxy.spaxel['specidx']] == pytest.approx(galaxy.spaxel['flux'], abs=abs)

    #@pytest.mark.xfail  # This test fails in some cases
    @pytest.mark.uses_web
    @pytest.mark.parametrize('monkeyconfig',
                             [('sasurl', 'http://www.averywrongurl.com')],
                             ids=['wrongurl'], indirect=True)
    def test_getSpaxel_remote_fail_badresponse(self, monkeyconfig):

        assert config.urlmap is not None

        with pytest.raises(MarvinError, match='Failed to establish a new connection'):
            Cube(mangaid='1-209232', mode='remote')

        #assert 'Failed to establish a new connection' in str(cm.value)

    @pytest.mark.parametrize('monkeyconfig',
                             [('release', 'DR17')],
                             ids=['dr17'], indirect=True)
    def test_getSpaxel_remote_drpver_differ_from_global(self, galaxy, monkeyconfig):
        if galaxy.release == 'DR17':
            pytest.skip('Skipping release for forced global DR17')

        assert config.release == 'DR17'

        cube = Cube(plateifu=galaxy.plateifu, mode='remote', release=galaxy.release)
        expected = galaxy.spaxel['flux']

        spectrum = cube.getSpaxel(ra=galaxy.spaxel['ra'], dec=galaxy.spaxel['dec']).flux
        assert spectrum.value[galaxy.spaxel['specidx']] == pytest.approx(expected)

    @pytest.mark.uses_web
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
        abs = 1.e-7

        assert spaxel_slice_file.flux.value[spec_idx] == pytest.approx(flux, abs=abs)
        assert spaxel_slice_db.flux.value[spec_idx] == pytest.approx(flux, abs=abs)
        assert spaxel_slice_api.flux.value[spec_idx] == pytest.approx(flux, abs=abs)

        assert spaxel_slice_file.flux.ivar[spec_idx] == pytest.approx(ivar)
        assert spaxel_slice_db.flux.ivar[spec_idx] == pytest.approx(ivar)
        assert spaxel_slice_api.flux.ivar[spec_idx] == pytest.approx(ivar)

        assert spaxel_slice_file.flux.mask[spec_idx] == pytest.approx(mask)
        assert spaxel_slice_db.flux.mask[spec_idx] == pytest.approx(mask)
        assert spaxel_slice_api.flux.mask[spec_idx] == pytest.approx(mask)

        xx_cen = galaxy.spaxel['x_cen']
        yy_cen = galaxy.spaxel['y_cen']

        try:
            spaxel_getspaxel_file = cube_file.getSpaxel(x=xx_cen, y=yy_cen)
            spaxel_getspaxel_db = cube_db.getSpaxel(x=xx_cen, y=yy_cen)
            spaxel_getspaxel_api = cube_api.getSpaxel(x=xx_cen, y=yy_cen)
        except MarvinError as ee:
            assert 'do not correspond to a valid binid' in str(ee)
            pytest.skip()

        assert spaxel_getspaxel_file.flux.value[spec_idx] == pytest.approx(flux, abs=1e-6)
        assert spaxel_getspaxel_db.flux.value[spec_idx] == pytest.approx(flux, abs=1e-6)
        assert spaxel_getspaxel_api.flux.value[spec_idx] == pytest.approx(flux, abs=1e-6)

        assert spaxel_getspaxel_file.flux.ivar[spec_idx] == pytest.approx(ivar)
        assert spaxel_getspaxel_db.flux.ivar[spec_idx] == pytest.approx(ivar)
        assert spaxel_getspaxel_api.flux.ivar[spec_idx] == pytest.approx(ivar)

        assert spaxel_getspaxel_file.flux.mask[spec_idx] == pytest.approx(mask)
        assert spaxel_getspaxel_db.flux.mask[spec_idx] == pytest.approx(mask)
        assert spaxel_getspaxel_api.flux.mask[spec_idx] == pytest.approx(mask)


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
            assert isinstance(spaxel, Spaxel)
        else:
            assert isinstance(spaxel, Spaxel)
            expected = galaxy.stellar_vel_ivar_x15_y8_lower[galaxy.release][galaxy.template.name]
            assert spaxel.maps_quantities['stellar_vel'].ivar == pytest.approx(expected, abs=1e-6)

        assert len(spaxel.maps_quantities.keys()) > 0

    def test_get_spaxel_test2(self, galaxy, data_origin):

        maps = Maps(**self._get_maps_kwargs(galaxy, data_origin))

        spaxel = _get_spaxel_helper(maps, 5, 5)

        if maps.is_binned():
            assert isinstance(spaxel, Spaxel)
        else:
            assert isinstance(spaxel, Spaxel)

        assert len(spaxel.maps_quantities.keys()) > 0

    def test_get_spaxel_no_db(self, galaxy, exporigin):
        """Tests getting an spaxel if there is no DB."""

        maps = Maps(**self._get_maps_kwargs(galaxy, exporigin))
        spaxel = _get_spaxel_helper(maps, 5, 5)

        assert spaxel.getMaps().data_origin == exporigin

        if maps.is_binned():
            assert isinstance(spaxel, Spaxel)
        else:
            assert isinstance(spaxel, Spaxel)

        assert len(spaxel.maps_quantities.keys()) > 0

    @marvin_test_if(mark='include', galaxy=dict(bintype=['HYB10']))
    def test_values(self, galaxy, exporigin):

        template = str(galaxy.template)

        if template not in galaxy.dap:
            pytest.skip()

        maps = Maps(**self._get_maps_kwargs(galaxy, exporigin))

        xx = galaxy.dap['x']
        yy = galaxy.dap['y']

        for channel in galaxy.dap[template]:

            if channel == 'model':
                continue

            channel_data = galaxy.dap[template][channel]
            map = maps[channel]

            assert map[yy, xx].value == pytest.approx(channel_data['value'], abs=2.e-4)
            assert map.unit.scale == 1e-17
            assert map.unit.to_string() == channel_data['unit']

            assert map[yy, xx].mask == pytest.approx(channel_data['mask'], abs=2.e-4)
            assert map[yy, xx].ivar == pytest.approx(channel_data['ivar'], abs=2.e-4)

    @marvin_test_if(mark='include', galaxy=dict(bintype=['HYB10']))
    def test_deprecated(self, galaxy, exporigin):

        if exporigin != 'db':
            pytest.skip()

        maps = Maps(**self._get_maps_kwargs(galaxy, exporigin))

        for old_arg in ['drp', 'model', 'models']:

            with pytest.raises(MarvinDeprecationError) as ee:
                kwargs = {old_arg: True}
                maps.getSpaxel(x=0, y=0, **kwargs)

            assert 'the {0} parameter has been deprecated.'.format(old_arg) in str(ee)


#@marvin_test_if_class(mark='skip', galaxy=dict(release=['DR17']))
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
        spaxel = _get_spaxel_helper(model_cube, 1, 2, maps=False, cube=False)
        self._test_getspaxel(spaxel, galaxy)
        assert isinstance(spaxel.getCube(), Cube)
        assert 'flux' not in spaxel.cube_quantities
        assert isinstance(spaxel.getMaps(), Maps)
        assert len(spaxel.maps_quantities) == 0

    def test_getspaxel_matches_file_db_remote(self, galaxy):

        if galaxy.bintype != 'SPX':
            pytest.skip()

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

        assert spaxel_getspaxel_file.binned_flux.value[idx] == pytest.approx(flux, abs=1e-6)
        assert spaxel_getspaxel_db.binned_flux.value[idx] == pytest.approx(flux, abs=1e-6)
        assert spaxel_getspaxel_api.binned_flux.value[idx] == pytest.approx(flux, abs=1e-6)

        assert spaxel_getspaxel_file.binned_flux.ivar[idx] == pytest.approx(ivar)
        assert spaxel_getspaxel_db.binned_flux.ivar[idx] == pytest.approx(ivar)
        assert spaxel_getspaxel_api.binned_flux.ivar[idx] == pytest.approx(ivar)

        assert spaxel_getspaxel_file.binned_flux.mask[idx] == pytest.approx(mask)
        assert spaxel_getspaxel_db.binned_flux.mask[idx] == pytest.approx(mask)
        assert spaxel_getspaxel_api.binned_flux.mask[idx] == pytest.approx(mask)
