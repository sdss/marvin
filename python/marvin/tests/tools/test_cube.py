#!/usr/bin/env python

import os
import re

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy import wcs

from marvin import config
from marvin.tools.cube import Cube
from marvin.core.core import DotableCaseInsensitive
from marvin.core.exceptions import MarvinError
from marvin.tests import skipIfNoDB


@pytest.fixture(scope='module')
def galaxy_cube(galaxy):
    galaxy.filename = os.path.realpath(galaxy.cubepath)
    return galaxy

@pytest.fixture(scope='module')
def cubeFromFile(galaxy):
    return Cube(filename=galaxy.cubepath)


class TestCube(object):


    # Tests for Cube Load by File
    def test_cube_loadfail(self):
        with pytest.raises(AssertionError) as cm:
            Cube()
        assert 'Enter filename, plateifu, or mangaid!' in str(cm.value)

    def test_cube_load_from_local_file_by_filename_success(self, galaxy_cube):
        cube = Cube(filename=galaxy_cube.filename)
        assert cube is not None
        assert galaxy_cube.filename == cube.filename

    def test_cube_load_from_local_file_by_filename_fail(self):
        config.use_sentry = False
        self.filename = 'not_a_filename.fits'
        with pytest.raises(MarvinError):
            Cube(filename=self.filename)

    @skipIfNoDB
    def test_cube_load_from_local_database_success(self, db, galaxy):
        """Tests for Cube Load by Database."""
        cube = Cube(mangaid=galaxy.mangaid)
        assert cube is not None
        assert galaxy.mangaid == cube.mangaid
        assert galaxy.plate == cube.plate
        assert galaxy.dec == cube.dec
        assert galaxy.ra == cube.ra

    def _load_from_db_fail(self, params, errMsg, errType=MarvinError):
        with pytest.raises(errType) as cm:
            Cube(**params)
        assert errMsg in str(cm.value)

    @skipIfNoDB
    @pytest.mark.skip('Test fails beacuse config.db = None does not disable the local DB, and '
                      'there is currently no way of doing so.')
    def test_cube_load_from_local_database_nodbconnected(self):
        # need to undo setting the config.db to None so that subsequent tests will pass
        # config.db = None
        params = {'mangaid': self.mangaid, 'mode': 'local'}
        errMsg = 'No db connected'
        self._load_from_db_fail(params, errMsg)

    @skipIfNoDB
    def test_cube_load_from_local_database_noresultsfound(self, db):
        params = {'plateifu': '8485-0923', 'mode': 'local'}
        errMsg = 'Could not retrieve cube for plate-ifu {0}: No Results Found'.format(
            params['plateifu'])
        self._load_from_db_fail(params, errMsg, errType=MarvinError)

    @skipIfNoDB
    def test_cube_load_from_local_database_otherexception(self, db):
        params = {'plateifu': '84.85-1901', 'mode': 'local'}
        errMsg = 'Could not retrieve cube for plate-ifu {0}: Unknown exception'.format(
            params['plateifu'])
        self._load_from_db_fail(params, errMsg, errType=MarvinError)

    def test_cube_flux_from_local_database(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu, mode='local')
        flux = cube.flux
        assert cube.data_origin == 'db'

        cubeFlux = fits.getdata(galaxy.cubepath)
        assert np.allclose(flux, cubeFlux)

    @pytest.mark.slow
    def test_cube_flux_from_api(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu, mode='remote')
        flux = cube.flux
        assert cube.data_origin == 'api'

        cubeFlux = fits.getdata(galaxy.cubepath)
        assert pytest.approx(flux, cubeFlux)

    def test_cube_remote_drpver_differ_from_global(self, galaxy):

        # This tests requires having the cube for 8485-1901 loaded for both
        # MPL-4 and MPL-5.

        config.setMPL('MPL-5')
        assert config.release == 'MPL-5'

        cube = Cube(plateifu=galaxy.plateifu, mode='remote', release='MPL-4')
        assert cube._drpver == 'v1_5_1'
        assert cube.header['VERSDRP3'].strip() == 'v1_5_0'

    @pytest.mark.parametrize('plateifu, filename, mode',
                             [(None, 'galaxy.cubepath', None),
                              ('galaxy.plateifu', None, 'local'),
                              ('galaxy.plateifu', None, 'remote')],
                             ids=('file', 'db', 'remote'))
    def test_cube_redshift(self, galaxy, plateifu, filename, mode):

        # TODO add 7443-12701 to local DB and remove this skip
        if ((galaxy.plateifu != '8485-1901') and (mode in [None, 'local']) and
                (config.db == 'local')):
            pytest.skip('Not the one true galaxy.')

        plateifu = eval(plateifu) if plateifu is not None else None
        filename = eval(filename) if filename is not None else None
        cube = Cube(plateifu=plateifu, filename=filename, mode=mode)
        assert pytest.approx(cube.nsa.z, galaxy.redshift)

    @pytest.mark.parametrize('plateifu, filename',
                             [(None, 'galaxy.cubepath'),
                              ('galaxy.plateifu', None)],
                             ids=('filename', 'plateifu'))
    @pytest.mark.parametrize('nsa_source',
                             ['auto', 'nsa', 'drpall'])
    @pytest.mark.parametrize('mode',
                             [None, 'remote'])
    def test_nsa_redshift(self, galaxy, plateifu, filename, nsa_source, mode):
        if (plateifu is None) and (filename is not None) and (mode == 'remote'):
            pytest.skip('filename not allowed in remote mode.')

        # TODO add 7443-12701 to local DB and remove this skip
        if (galaxy.plateifu != '8485-1901') and (mode is None) and (config.db == 'local'):
            pytest.skip('Not the one true galaxy.')

        plateifu = eval(plateifu) if plateifu is not None else None
        filename = eval(filename) if filename is not None else None
        cube = Cube(plateifu=plateifu, filename=filename, nsa_source=nsa_source, mode=mode)
        assert cube.nsa_source == nsa_source
        assert cube.nsa['nsaid'] == galaxy.nsaid
        assert isinstance(cube.nsa, DotableCaseInsensitive)
        if mode == 'drpall':
            assert 'profmean_ivar' not in cube.nsa.keys()
        assert 'zdist' in cube.nsa.keys()
        assert pytest.approx(cube.nsa['zdist'], galaxy.redshift)
        assert pytest.approx(cube.nsa['sersic_flux_ivar'][0], galaxy.nsa_sersic_flux_ivar0)

    def test_release(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        assert cube.release == galaxy.release

    def test_set_release_fails(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu)
        with pytest.raises(MarvinError) as ee:
            cube.release = 'a'
            assert 'the release cannot be changed' in str(ee.exception)

    # TODO remove because added 7443-12701 to NSA
    def test_load_7443_12701_file(self, galaxy):
        """Loads a cube that is not in the NSA catalogue."""

        config.setMPL('MPL-5')
        galaxy.set_filepaths()
        filename = os.path.realpath(os.path.join(
            galaxy.drppath, '7443/stack/manga-7443-12701-LOGCUBE.fits.gz'))
        cube = Cube(filename=filename)
        assert cube.data_origin == 'file'
        assert 'elpetro_amivar' in cube.nsa

    # TODO remove because added 7443-12701 to NSA
    def test_load_7443_12701_db(self):
        """Loads a cube that is not in the NSA catalogue."""

        self._update_release('MPL-5')
        cube = Cube(plateifu='7443-12701')
        assert cube.data_origin == 'db'
        assert cube.nsa is None

    # TODO remove because added 7443-12701 to NSA
    def test_load_7443_12701_api(self):
        """Loads a cube that is not in the NSA catalogue."""

        self._update_release('MPL-5')
        cube = Cube(plateifu='7443-12701', mode='remote')
        assert cube.data_origin == 'api'
        assert cube.nsa is None

    def test_load_filename_does_not_exist(self):
        """Tries to load a file that does not exist, in auto mode."""
        config.mode = 'auto'
        with pytest.raises(MarvinError) as ee:
            Cube(filename='hola.fits')

        assert re.match(r'input file .+hola.fits not found', str(ee.value)) is not None

    def test_load_filename_remote(self):
        """Tries to load a filename in remote mode and fails."""
        config.mode = 'remote'
        with pytest.raises(MarvinError) as ee:
            Cube(filename='hola.fits')

        assert 'filename not allowed in remote mode' in str(ee.value)


class TestGetSpaxel(object):

    #  Tests for getSpaxel
    def _test_getSpaxel(self, cube, expected, **kwargs):
        """Convenience method to test getSpaxel."""
        ext = kwargs.pop('ext', 'flux')
        idx = kwargs.pop('idx')
        spectrum = cube.getSpaxel(**kwargs).spectrum
        assert pytest.approx(getattr(spectrum, ext)[idx], expected)
    
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
    def test_getSpaxel_inputs(self, cubeFromFile, x, y, ra, dec, excType, message):
        """Tests exceptions when getSpaxel gets inappropriate inputs."""
        kwargs = self._dropNones(x=x, y=y, ra=ra, dec=dec)

        with pytest.raises(excType) as ee:
            cubeFromFile.getSpaxel(**kwargs)

        assert message in str(ee.value)

    @pytest.mark.parametrize('x, y, ra, dec, xyorig, idx, mpl4, mpl5',
                             [(10, 5, None, None, None, 10, -0.062497504, -0.063987106),
                              (10, 5, None, None, 'lower', 3000, 0.017929086, 0.017640527),
                              (0, 0, None, None, None, 3000, 1.0493046, 1.0505702),
                              (0, 0, None, None, 'lower', 3000, 0., 0.),
                              (None, None, 232.5443, 48.6899, None, 3000, 0.62007582, 0.6171338),
                              (None, None, 232.544279, 48.6899232, None, 3000, 0.62007582, 0.6171338),
                              (None, None, 232.54, 48.69, None, 3000, MarvinError, MarvinError),
                              (None, None, 233, 49, None, 3000, MarvinError, MarvinError)
                              ],
                             ids=['xy', 'xylower', 'x0y0', 'x0y0lower', 'radec-partial',
                                  'radec-full', 'radec-fail-twosigfig', 'radec-fail-int'])
    @pytest.mark.parametrize('data_origin', ['db', 'file', 'api'])
    def test_getSpaxel_file_flux(self, request, galaxy, data_origin, x, y, ra, dec, xyorig, idx,
                                 mpl4, mpl5):
        if data_origin == 'db':
            db = request.getfixturevalue('db')
            if db.session is None:
                pytest.skip('Skip because no DB.')
            cube = Cube(mangaid=galaxy.mangaid)
        elif data_origin == 'file':
            cube = request.getfixturevalue('cubeFromFile')
        else:
            cube = Cube(mangaid=galaxy.mangaid, mode='remote')

        kwargs = self._dropNones(x=x, y=y, ra=ra, dec=dec, xyorig=xyorig)
        ext = kwargs.pop('ext', 'flux')
        mpls = {'MPL-4': mpl4, 'MPL-5': mpl5}
        expected = mpls[config.release]

        if expected is MarvinError:
            with pytest.raises(MarvinError) as cm:
                spectrum = cube.getSpaxel(**kwargs).spectrum
                actual = getattr(spectrum, ext)[idx]
            assert 'some indices are out of limits.' in str(cm.value)
        else:
            spectrum = cube.getSpaxel(**kwargs).spectrum
            assert pytest.approx(getattr(spectrum, ext)[idx], expected)


    def _test_getSpaxel_remote(self, specIndex, expect, **kwargs):
        """Tests for getSpaxel remotely."""

        cube = Cube(mangaid=self.mangaid, mode='remote')
        self._test_getSpaxel(cube, specIndex, expect, **kwargs)

    def test_getSpaxel_remote_x_y_success(self):

        expect = -0.062497499999999997
        self._test_getSpaxel_remote(10, expect, x=10, y=5)

    def test_getSpaxel_remote_ra_dec_success(self):

        expect = 0.62007582
        self._test_getSpaxel_remote(3000, expect, ra=232.544279, dec=48.6899232)

    def _getSpaxel_remote_fail(self, ra, dec, errMsg1, errMsg2=None, excType=MarvinError):

        cube = Cube(mangaid=self.mangaid, mode='remote')

        with pytest.raises(excType) as cm:
            cube.getSpaxel(ra=ra, dec=dec)

        assert errMsg1 in str(cm.exception)
        if errMsg2:
            assert errMsg2 in str(cm.exception)

    def test_getSpaxel_remote_fail_badresponse(self):

        config.sasurl = 'http://www.averywrongurl.com'
        assert config.urlmap is not None

        with pytest.raises(MarvinError) as cm:
            Cube(mangaid=self.mangaid, mode='remote')

        assert 'Failed to establish a new connection' in str(cm.exception)

    def test_getSpaxel_remote_fail_badpixcoords(self):

        assert config.urlmap is not None
        self._getSpaxel_remote_fail(232, 48, 'some indices are out of limits.')

    def _test_getSpaxel_array(self, cube, nCoords, specIndex, expected, **kwargs):
        """Tests getSpaxel with array coordinates."""

        spaxels = cube.getSpaxel(**kwargs)

        assert len(spaxels) == nCoords
        fluxes = np.array([spaxel.spectrum.flux for spaxel in spaxels])

        assert_allclose(fluxes[:, specIndex], expected, rtol=1e-6)

    def test_getSpaxel_file_flux_x_y_lower_array(self):

        x = [10, 0]
        y = [5, 0]
        expected = [0.017929086, 0.0]

        cube = self.cubeFromFile
        self._test_getSpaxel_array(cube, 2, 3000, expected,
                                   x=x, y=y, xyorig='lower')

    def test_getSpaxel_db_flux_x_y_lower_array(self):

        x = [10, 0]
        y = [5, 0]
        expected = [0.017929086, 0.0]

        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel_array(cube, 2, 3000, expected,
                                   x=x, y=y, xyorig='lower')

    def test_getSpaxel_remote_flux_x_y_lower_array(self):

        x = [10, 0]
        y = [5, 0]
        expected = [0.017929086, 0.0]

        cube = Cube(mangaid=self.mangaid, mode='remote')
        self._test_getSpaxel_array(cube, 2, 3000, expected,
                                   x=x, y=y, xyorig='lower')

    def test_getSpaxel_file_flux_ra_dec_lower_array(self):

        ra = [232.546173, 232.548277]
        dec = [48.6885343, 48.6878398]
        expected = [0.017929086, 0.0]

        cube = self.cubeFromFile
        self._test_getSpaxel_array(cube, 2, 3000, expected,
                                   ra=ra, dec=dec, xyorig='lower')

    def test_getSpaxel_db_flux_ra_dec_lower_array(self):

        ra = [232.546173, 232.548277]
        dec = [48.6885343, 48.6878398]
        expected = [0.017929086, 0.0]

        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel_array(cube, 2, 3000, expected,
                                   ra=ra, dec=dec, xyorig='lower')

    def test_getSpaxel_remote_flux_ra_dec_lower_array(self):

        ra = [232.546173, 232.548277]
        dec = [48.6885343, 48.6878398]
        expected = [0.017929086, 0.0]

        cube = Cube(mangaid=self.mangaid, mode='remote')
        self._test_getSpaxel_array(cube, 2, 3000, expected,
                                   ra=ra, dec=dec, xyorig='lower')

    def test_getSpaxel_global_xyorig_center(self):
        config.xyorig = 'center'
        expect = -0.062497499999999997
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel(cube, 10, expect, x=10, y=5)

    def test_getSpaxel_global_xyorig_lower(self):
        config.xyorig = 'lower'
        expect = 0.017929086
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel(cube, 3000, expect, x=10, y=5)

    def test_getSpaxel_remote_drpver_differ_from_global(self):

        self._update_release('MPL-5')
        assert config.release == 'MPL-5'

        cube = Cube(plateifu=self.plateifu, mode='remote', release='MPL-4')
        expect = 0.62007582
        self._test_getSpaxel(cube, 3000, expect, ra=232.544279, dec=48.6899232)

    @skipIfNoDB
    def test_getspaxel_matches_file_db_remote(self):

        self._update_release('MPL-4')
        assert config.release == 'MPL-4'

        cube_file = Cube(filename=self.filename)
        cube_db = Cube(plateifu=self.plateifu)
        cube_api = Cube(plateifu=self.plateifu, mode='remote')

        assert cube_file.data_origin == 'file'
        assert cube_db.data_origin == 'db'
        assert cube_api.data_origin == 'api'

        xx = 12
        yy = 5
        spec_idx = 200

        spaxel_slice_file = cube_file[yy, xx]
        spaxel_slice_db = cube_db[yy, xx]
        spaxel_slice_api = cube_api[yy, xx]

        flux_result = 0.017639931
        ivar_result = 352.12421
        mask_result = 1026

        assert round(abs(spaxel_slice_file.spectrum.flux[spec_idx]-flux_result), 7) == 0
        assert round(abs(spaxel_slice_db.spectrum.flux[spec_idx]-flux_result), 7) == 0
        assert round(abs(spaxel_slice_api.spectrum.flux[spec_idx]-flux_result), 7) == 0

        assert round(abs(spaxel_slice_file.spectrum.ivar[spec_idx]-ivar_result), 5) == 0
        assert round(abs(spaxel_slice_db.spectrum.ivar[spec_idx]-ivar_result), 3) == 0
        assert round(abs(spaxel_slice_api.spectrum.ivar[spec_idx]-ivar_result), 3) == 0

        assert round(abs(spaxel_slice_file.spectrum.mask[spec_idx]-mask_result), 7) == 0
        assert round(abs(spaxel_slice_db.spectrum.mask[spec_idx]-mask_result), 7) == 0
        assert round(abs(spaxel_slice_api.spectrum.mask[spec_idx]-mask_result), 7) == 0

        xx_cen = -5
        yy_cen = -12

        spaxel_getspaxel_file = cube_file.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_db = cube_db.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_api = cube_api.getSpaxel(x=xx_cen, y=yy_cen)

        assert round(abs(spaxel_getspaxel_file.spectrum.flux[spec_idx]-flux_result), 7) == 0
        assert round(abs(spaxel_getspaxel_db.spectrum.flux[spec_idx]-flux_result), 7) == 0
        assert round(abs(spaxel_getspaxel_api.spectrum.flux[spec_idx]-flux_result), 7) == 0

        assert round(abs(spaxel_getspaxel_file.spectrum.ivar[spec_idx]-ivar_result), 5) == 0
        assert round(abs(spaxel_getspaxel_db.spectrum.ivar[spec_idx]-ivar_result), 3) == 0
        assert round(abs(spaxel_getspaxel_api.spectrum.ivar[spec_idx]-ivar_result), 3) == 0

        assert round(abs(spaxel_getspaxel_file.spectrum.mask[spec_idx]-mask_result), 7) == 0
        assert round(abs(spaxel_getspaxel_db.spectrum.mask[spec_idx]-mask_result), 7) == 0
        assert round(abs(spaxel_getspaxel_api.spectrum.mask[spec_idx]-mask_result), 7) == 0


class TestWCS(object):

    def test_wcs_file(self):
        cube = Cube(filename=self.filename)
        assert isinstance(cube.wcs, wcs.WCS)
        assert round(abs(cube.wcs.wcs.cd[1, 1]-0.000138889), 7) == 0

    def test_wcs_db(self):
        cube = Cube(plateifu=self.plateifu)
        assert isinstance(cube.wcs, wcs.WCS)
        assert round(abs(cube.wcs.wcs.cd[1, 1]-0.000138889), 7) == 0

    def test_wcs_api(self):
        cube = Cube(plateifu=self.plateifu, mode='remote')
        assert isinstance(cube.wcs, wcs.WCS)
        assert round(abs(cube.wcs.wcs.pc[1, 1]-0.000138889), 7) == 0


class TestPickling(object):

    def setUp(self):
        super(TestPickling, self).setUp()
        self._files_created = []

    def tearDown(self):

        super(TestPickling, self).tearDown()

        for fp in self._files_created:
            if os.path.exists(fp):
                os.remove(fp)

    def test_pickling_file(self):

        cube = Cube(filename=self.filename)
        assert cube.data_origin == 'file'
        assert isinstance(cube, Cube)
        assert cube.data is not None

        path = cube.save()
        self._files_created.append(path)

        assert os.path.exists(path)
        assert os.path.realpath(path) == os.path.realpath(self.filename[0:-7] + 'mpf')
        assert cube.data is not None

        cube = None
        assert cube is None

        cube_restored = Cube.restore(path)
        assert cube_restored.data_origin == 'file'
        assert isinstance(cube_restored, Cube)
        assert cube_restored.data is not None

    def test_pickling_file_custom_path(self):

        cube = Cube(filename=self.filename)

        test_path = '~/test.mpf'
        path = cube.save(path=test_path)
        self._files_created.append(path)

        assert os.path.exists(path)
        assert path == os.path.realpath(os.path.expanduser(test_path))

        cube_restored = Cube.restore(path, delete=True)
        assert cube_restored.data_origin == 'file'
        assert isinstance(cube_restored, Cube)
        assert cube_restored.data is not None

        assert not os.path.exists(path)

    def test_pickling_db(self):

        cube = Cube(plateifu=self.plateifu)
        assert cube.data_origin == 'db'

        with pytest.raises(MarvinError) as ee:
            cube.save()

        assert 'objects with data_origin=\'db\' cannot be saved.' in str(ee.exception)

    def test_pickling_api(self):

        cube = Cube(plateifu=self.plateifu, mode='remote')
        assert cube.data_origin == 'api'
        assert isinstance(cube, Cube)
        assert cube.data is None

        path = cube.save()
        self._files_created.append(path)

        assert os.path.exists(path)
        assert os.path.realpath(path) == os.path.realpath(self.filename[0:-7] + 'mpf')

        cube = None
        assert cube is None

        cube_restored = Cube.restore(path)
        assert cube_restored.data_origin == 'api'
        assert isinstance(cube_restored, Cube)
        assert cube_restored.data is None
        assert cube_restored.header['VERSDRP3'] == 'v1_5_0'
