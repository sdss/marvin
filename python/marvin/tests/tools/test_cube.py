#!/usr/bin/env python

import os
import re

import pytest
import numpy as np
from astropy.io import fits
from astropy import wcs

from marvin import config, marvindb
from marvin.tools.cube import Cube
from marvin.core.core import DotableCaseInsensitive
from marvin.core.exceptions import MarvinError
from marvin.tests import skipIfNoDB, marvin_test_if


@pytest.fixture(autouse=True)
def skipbins(galaxy):
    if galaxy.bintype not in ['SPX', 'NONE']:
        pytest.skip('Skipping all bins for Cube tests')
    if galaxy.template not in ['MILES-THIN', 'GAU-MILESHC']:
        pytest.skip('Skipping all templates for Cube tests')


class TestCube(object):

    def test_cube_loadfail(self):
        with pytest.raises(AssertionError) as cm:
            Cube()
        assert 'Enter filename, plateifu, or mangaid!' in str(cm.value)

    def test_cube_load_from_local_file_by_filename_success(self, galaxy):
        cube = Cube(filename=galaxy.cubepath)
        assert cube is not None
        assert os.path.realpath(galaxy.cubepath) == cube.filename

    def test_cube_load_from_local_file_by_filename_fail(self):
        with pytest.raises(MarvinError):
            Cube(filename='not_a_filename.fits')

    def test_cube_load_from_local_database_success(self, galaxy):
        """Tests for Cube Load by Database."""
        cube = Cube(mangaid=galaxy.mangaid)
        assert cube is not None
        assert galaxy.mangaid == cube.mangaid
        assert galaxy.plate == cube.plate
        assert galaxy.dec == cube.dec
        assert galaxy.ra == cube.ra

    @pytest.mark.parametrize('plateifu, mode, errmsg',
                             [('8485-0923', 'local', 'Could not retrieve cube for plate-ifu 8485-0923: No Results Found')],
                             ids=['noresults'])
    def test_cube_from_db_fail(self, plateifu, mode, errmsg):
        with pytest.raises(MarvinError) as cm:
            c = Cube(plateifu=plateifu, mode=mode)
        assert errmsg in str(cm.value)

    # @pytest.mark.slow
    @marvin_test_if(mark='include', cube={'plateifu': '8485-1901'})
    def test_cube_flux(self, cube):
        assert cube.flux is not None
        assert isinstance(cube.flux, np.ndarray)

    @pytest.mark.parametrize('monkeyconfig',
                             [('release', 'MPL-5')],
                             ids=['mpl5'], indirect=True)
    def test_cube_remote_drpver_differ_from_global(self, galaxy, monkeyconfig):

        assert config.release == 'MPL-5'
        cube = Cube(plateifu=galaxy.plateifu, mode='remote', release='MPL-4')
        assert cube._drpver == 'v1_5_1'
        assert cube.header['VERSDRP3'].strip() == 'v1_5_0'

    def test_cube_redshift(self, cube, galaxy):
        assert cube.data_origin == cube.exporigin
        redshift = cube.nsa.redshift if cube.release == 'MPL-4' and cube.data_origin == 'file' else cube.nsa.z
        assert pytest.approx(redshift, galaxy.redshift)

    def test_release(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        assert cube.release == galaxy.release

    def test_set_release_fails(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        with pytest.raises(MarvinError) as ee:
            cube.release = 'a'
        assert 'the release cannot be changed' in str(ee.value)

    def test_load_filename_does_not_exist(self):
        """Tries to load a file that does not exist, in auto mode."""
        with pytest.raises(MarvinError) as ee:
            Cube(filename='hola.fits', mode='auto')

        assert re.match(r'input file .+hola.fits not found', str(ee.value)) is not None

    def test_load_filename_remote(self):
        """Tries to load a filename in remote mode and fails."""
        with pytest.raises(MarvinError) as ee:
            Cube(filename='hola.fits', mode='remote')

        assert 'filename not allowed in remote mode' in str(ee.value)


class TestGetSpaxel(object):

    def _dropNones(self, **kwargs):
        for k, v in list(kwargs.items()):
            if v is None:
                del kwargs[k]
        return kwargs

    @pytest.mark.parametrize('x, y, ra, dec, excType, message',
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
        flux = spaxel.spectrum.flux
        assert pytest.approx(flux[galaxy.spaxel['specidx']], galaxy.spaxel['flux'])

    @pytest.mark.parametrize('monkeyconfig',
                             [('sasurl', 'http://www.averywrongurl.com')],
                             ids=['wrongurl'], indirect=True)
    def test_getSpaxel_remote_fail_badresponse(self, monkeyconfig):
        assert config.urlmap is not None

        with pytest.raises(MarvinError) as cm:
            Cube(mangaid='1-209232', mode='remote')

        assert 'Failed to establish a new connection' in str(cm.value)

    def test_getSpaxel_remote_drpver_differ_from_global(self, galaxy):
        config.setMPL('MPL-5')
        assert config.release == 'MPL-5'

        cube = Cube(plateifu=galaxy.plateifu, mode='remote', release='MPL-4')
        expected = galaxy.spaxel['flux']

        spectrum = cube.getSpaxel(ra=galaxy.spaxel['ra'], dec=galaxy.spaxel['dec']).spectrum
        assert pytest.approx(spectrum.flux[galaxy.spaxel['specidx']], expected)

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

        assert pytest.approx(spaxel_slice_file.spectrum.flux[spec_idx], flux)
        assert pytest.approx(spaxel_slice_db.spectrum.flux[spec_idx], flux)
        assert pytest.approx(spaxel_slice_api.spectrum.flux[spec_idx], flux)

        assert pytest.approx(spaxel_slice_file.spectrum.ivar[spec_idx], ivar)
        assert pytest.approx(spaxel_slice_db.spectrum.ivar[spec_idx], ivar)
        assert pytest.approx(spaxel_slice_api.spectrum.ivar[spec_idx], ivar)

        assert pytest.approx(spaxel_slice_file.spectrum.mask[spec_idx], mask)
        assert pytest.approx(spaxel_slice_db.spectrum.mask[spec_idx], mask)
        assert pytest.approx(spaxel_slice_api.spectrum.mask[spec_idx], mask)

        xx_cen = galaxy.spaxel['x_cen']
        yy_cen = galaxy.spaxel['y_cen']

        spaxel_getspaxel_file = cube_file.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_db = cube_db.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_api = cube_api.getSpaxel(x=xx_cen, y=yy_cen)

        assert pytest.approx(spaxel_getspaxel_file.spectrum.flux[spec_idx], flux)
        assert pytest.approx(spaxel_getspaxel_db.spectrum.flux[spec_idx], flux)
        assert pytest.approx(spaxel_getspaxel_api.spectrum.flux[spec_idx], flux)

        assert pytest.approx(spaxel_getspaxel_file.spectrum.ivar[spec_idx], ivar)
        assert pytest.approx(spaxel_getspaxel_db.spectrum.ivar[spec_idx], ivar)
        assert pytest.approx(spaxel_getspaxel_api.spectrum.ivar[spec_idx], ivar)

        assert pytest.approx(spaxel_getspaxel_file.spectrum.mask[spec_idx], mask)
        assert pytest.approx(spaxel_getspaxel_db.spectrum.mask[spec_idx], mask)
        assert pytest.approx(spaxel_getspaxel_api.spectrum.mask[spec_idx], mask)


class TestWCS(object):

    def test_wcs(self, cube):
        assert cube.data_origin == cube.exporigin
        assert isinstance(cube.wcs, wcs.WCS)
        comp = cube.wcs.wcs.pc if cube.data_origin == 'api' else cube.wcs.wcs.cd
        assert pytest.approx(comp[1, 1], 0.000138889)


class TestPickling(object):

    def test_pickling_file(self, temp_scratch, galaxy):
        cube = Cube(filename=galaxy.cubepath)
        assert cube.data_origin == 'file'
        assert isinstance(cube, Cube)
        assert cube.data is not None

        assert not os.path.isfile(galaxy.cubepath[0:-7] + 'mpf')
        cube_file = temp_scratch.join('test_cube.mpf')
        path = cube.save(str(cube_file))
        assert cube_file.check() is True
        assert cube.data is not None

        cube = None
        assert cube is None

        cube_restored = Cube.restore(str(cube_file))
        assert cube_restored.data_origin == 'file'
        assert isinstance(cube_restored, Cube)
        assert cube_restored.data is not None

    def test_pickling_file_custom_path(self, temp_scratch, galaxy):
        cube = Cube(filename=galaxy.cubepath)
        assert cube.data_origin == 'file'
        assert isinstance(cube, Cube)
        assert cube.data is not None

        test_path = temp_scratch.join('cubepickle').join('test_cube.mpf')
        assert test_path.check(file=1) is False

        path = cube.save(path=str(test_path))
        assert test_path.check(file=1) is True
        assert path == os.path.realpath(os.path.expanduser(str(test_path)))

        cube_restored = Cube.restore(str(test_path), delete=True)
        assert cube_restored.data_origin == 'file'
        assert isinstance(cube_restored, Cube)
        assert cube_restored.data is not None

        assert not os.path.exists(path)

    def test_pickling_db(self, galaxy, temp_scratch):
        cube = Cube(plateifu=galaxy.plateifu)
        assert cube.data_origin == 'db'

        file = temp_scratch.join('test_cube_db.mpf')
        with pytest.raises(MarvinError) as cm:
            cube.save(str(file))

        assert 'objects with data_origin=\'db\' cannot be saved.' in str(cm.value)

    def test_pickling_api(self, temp_scratch, galaxy):
        cube = Cube(plateifu=galaxy.plateifu, mode='remote')
        assert cube.data_origin == 'api'
        assert isinstance(cube, Cube)
        assert cube.data is None

        test_path = temp_scratch.join('test_cube_api.mpf')

        path = cube.save(str(test_path))
        assert test_path.check() is True

        cube = None
        assert cube is None

        cube_restored = Cube.restore(str(test_path))
        assert cube_restored.data_origin == 'api'
        assert isinstance(cube_restored, Cube)
        assert cube_restored.data is None
