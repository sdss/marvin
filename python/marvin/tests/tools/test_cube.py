#!/usr/bin/env python

import os

import pytest

import numpy as np
from numpy.testing import assert_allclose

from astropy.io import fits
from astropy import wcs

from marvin import config
from marvin.tools.cube import Cube
from marvin.core.core import DotableCaseInsensitive
from marvin.core.exceptions import MarvinError
from marvin.tests import skipIfNoDB, slow


@pytest.fixture(scope='module')
def galaxy_cube(galaxy):
    galaxy.filename = os.path.realpath(galaxy.cubepath)
    return galaxy


class TestCube(object):

    # def test_mpl_version(self):
    #     assert config.release == self.outrelease

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
        
        # if not db.session:
        #     pytest.skip("Skipped because no DB.")

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
    @pytest.mark.skip('Test fails beacuse config.db = None does not disable the local DB, and there'
                      ' is currently no way of doing so.')
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

    @slow
    def test_cube_flux_from_api(self, galaxy):
        cube = Cube(plateifu=galaxy.plateifu, mode='remote')
        flux = cube.flux
        assert cube.data_origin == 'api'

        cubeFlux = fits.getdata(galaxy.cubepath)
        assert pytest.approx(flux, cubeFlux)

    def test_cube_remote_drpver_differ_from_global(self):

        # This tests requires having the cube for 8485-1901 loaded for both
        # MPL-4 and MPL-5.

        self._update_release('MPL-5')
        assert config.release == 'MPL-5'

        cube = Cube(plateifu=self.plateifu, mode='remote', release='MPL-4')
        assert cube._drpver == 'v1_5_1'
        assert cube.header['VERSDRP3'].strip() == 'v1_5_0'

    def test_cube_file_redshift(self):
        cube = Cube(filename=self.filename)
        assert round(abs(cube.nsa.redshift-0.0407447), 7) == 0

    def test_cube_db_redshift(self):
        cube = Cube(plateifu=self.plateifu, mode='local')
        assert round(abs(cube.nsa.z-0.0407447), 7) == 0

    def test_cube_remote_redshift(self):
        cube = Cube(plateifu=self.plateifu, mode='remote')
        assert round(abs(cube.nsa.z-0.0407447), 7) == 0

    def _test_nsa(self, nsa_data, mode='nsa'):
        assert isinstance(nsa_data, DotableCaseInsensitive)
        if mode == 'drpall':
            assert 'profmean_ivar' not in nsa_data.keys()
        assert 'zdist' in nsa_data.keys()
        assert round(abs(nsa_data['zdist']-0.041201399999999999), 7) == 0

    def test_nsa_file_auto(self):
        cube = Cube(filename=self.filename)
        assert cube.nsa_source == 'auto'
        self._test_nsa(cube.nsa)

    def test_nsa_file_nsa(self):
        cube = Cube(filename=self.filename, nsa_source='nsa')
        assert cube.nsa_source == 'nsa'
        self._test_nsa(cube.nsa)

    def test_nsa_file_drpall(self):
        cube = Cube(plateifu=self.plateifu, nsa_source='drpall')
        assert cube.nsa_source == 'drpall'
        self._test_nsa(cube.nsa)

    def test_nsa_db_auto(self):
        # import pdb; pdb.set_trace()
        cube = Cube(plateifu=self.plateifu)
        assert cube.nsa_source == 'auto'
        self._test_nsa(cube.nsa)

    def test_nsa_db_nsa(self):
        cube = Cube(plateifu=self.plateifu, nsa_source='nsa')
        assert cube.nsa_source == 'nsa'
        self._test_nsa(cube.nsa)

    def test_nsa_db_drpall(self):
        cube = Cube(plateifu=self.plateifu, nsa_source='drpall')
        assert cube.nsa_source == 'drpall'
        self._test_nsa(cube.nsa)

    def test_nsa_remote_auto(self):
        cube = Cube(plateifu=self.plateifu, mode='remote')
        assert cube.nsa_source == 'auto'
        self._test_nsa(cube.nsa)

    def test_nsa_remote_nsa(self):
        cube = Cube(plateifu=self.plateifu, mode='remote', nsa_source='nsa')
        assert cube.nsa_source == 'nsa'
        self._test_nsa(cube.nsa)

    def test_nsa_remote_drpall(self):
        cube = Cube(plateifu=self.plateifu, mode='remote', nsa_source='drpall')
        assert cube.nsa_source == 'drpall'
        self._test_nsa(cube.nsa)

    def test_nsa_12_nsa(self):
        cube = Cube(mangaid='12-98126', nsa_source='nsa')
        self.assertEqual(cube.nsa_source, 'nsa')
        self.assertEqual(cube.nsa['nsaid'], 341153)
        self.assertAlmostEqual(cube.nsa['sersic_flux_ivar'][0], 0.046455454081)

    def test_release(self):
        cube = Cube(plateifu=self.plateifu)
        assert cube.release == 'MPL-4'

    def test_set_release_fails(self):
        cube = Cube(plateifu=self.plateifu)
        with pytest.raises(MarvinError) as ee:
            cube.release = 'a'
            assert 'the release cannot be changed' in str(ee.exception)

    def test_load_7443_12701_file(self):
        """Loads a cube that is not in the NSA catalogue."""

        self._update_release('MPL-5')
        self.set_filepaths()
        filename = os.path.realpath(os.path.join(
            self.drppath, '7443/stack/manga-7443-12701-LOGCUBE.fits.gz'))
        cube = Cube(filename=filename)
        assert cube.data_origin == 'file'
        assert 'elpetro_amivar' in cube.nsa

    def test_load_7443_12701_db(self):
        """Loads a cube that is not in the NSA catalogue."""

        self._update_release('MPL-5')
        cube = Cube(plateifu='7443-12701')
        assert cube.data_origin == 'db'
        assert cube.nsa is None

    def test_load_7443_12701_api(self):
        """Loads a cube that is not in the NSA catalogue."""

        self._update_release('MPL-5')
        cube = Cube(plateifu='7443-12701', mode='remote')
        assert cube.data_origin == 'api'
        assert cube.nsa is None

    def test_load_filename_does_not_exist(self):
        """Tries to load a file that does not exists, in auto mode."""

        config.mode = 'auto'

        with self.assertRaises(MarvinError) as ee:
            Cube(filename='hola.fits')

        self.assertIsNotNone(re.match(r'input file .+hola.fits not found', str(ee.exception)))

    def test_load_filename_remote(self):
        """Tries to load a filename in remote mode and fails."""

        config.mode = 'remote'

        with self.assertRaises(MarvinError) as ee:
            Cube(filename='hola.fits')

        self.assertIn('filename not allowed in remote mode', str(ee.exception))


class TestGetSpaxel(object):

    #  Tests for getSpaxel
    def _test_getSpaxel(self, cube, idx, expect, **kwargs):
        """Convenience method to test getSpaxel."""

        ext = kwargs.pop('ext', 'flux')
        spectrum = cube.getSpaxel(**kwargs).spectrum
        assert round(abs(getattr(spectrum, ext)[idx]-expect), 5) == 0

    def _test_getSpaxel_raise_exception(self, message, excType=AssertionError, **kwargs):
        """Convenience method to test exceptions raised by getSpaxel."""

        with pytest.raises(excType) as ee:
            self.cubeFromFile.getSpaxel(**kwargs)

        assert message in str(ee.exception)

    def test_getSpaxel_inputs(self):
        """Tests exceptions when getSpaxel gets inappropriate inputs."""

        self._test_getSpaxel_raise_exception(
            'Either use (x, y) or (ra, dec)', x=1, ra=1)

        self._test_getSpaxel_raise_exception(
            'Either use (x, y) or (ra, dec)', x=1, dec=1, ra=1)

        self._test_getSpaxel_raise_exception('Specify both x and y', x=1)

        self._test_getSpaxel_raise_exception('Specify both ra and dec', ra=1)

        self._test_getSpaxel_raise_exception(
            'You need to specify either (x, y) or (ra, dec)',
            excType=ValueError)

    def test_getSpaxel_outside_cube(self):
        """Tests getSpaxel when the input coords are outside the cube."""

        for xTest, yTest in [(-50, 1), (50, 1), (1, -50), (1, 50)]:
            self._test_getSpaxel_raise_exception(
                'some indices are out of limits.', x=xTest, y=yTest,
                excType=MarvinError)

        for raTest, decTest in [(1., 1.), (100, 60),
                                (232.546383, 1.), (1., 48.6883954)]:
            self._test_getSpaxel_raise_exception(
                'some indices are out of limits.', ra=raTest, dec=decTest,
                excType=MarvinError)

    def test_getSpaxel_file_flux_x_y(self):
        """Tests getSpaxel from a file cube with x, y inputs."""

        expect = -0.062497504
        self._test_getSpaxel(self.cubeFromFile, 10, expect, x=10, y=5)

    def test_getSpaxel_file_flux_x_y_lower(self):
        """Tests getSpaxel from a file with x, y inputs, xyorig=lower."""

        expect = 0.017929086
        self._test_getSpaxel(self.cubeFromFile, 3000, expect, x=10, y=5, xyorig='lower')

    def test_getSpaxel_file_flux_x_0_y_0(self):
        expect = 1.0493046
        self._test_getSpaxel(self.cubeFromFile, 3000, expect, x=0, y=0)

    def test_getSpaxel_file_flux_x_0_y_0_lower(self):
        expect = 0.0
        self._test_getSpaxel(self.cubeFromFile, 3000, expect, x=0, y=0, xyorig='lower')

    def _getSpaxel_file_flux_ra_dec(self, ra, dec):
        """Tests getSpaxel from a file cube with ra, dec inputs."""

        expect = 0.62007582
        self._test_getSpaxel(self.cubeFromFile, 3000, expect, ra=ra, dec=dec)

    def _getSpaxel_file_fail(self, ra, dec, errMsg):
        expect = 0.62007582
        with pytest.raises(MarvinError) as cm:
            self._test_getSpaxel(self.cubeFromFile, 3000, expect, ra=ra, dec=dec)
        assert errMsg in str(cm.exception)

    def test_getSpaxel_file_flux_ra_dec_full(self):
        self._getSpaxel_file_flux_ra_dec(ra=232.544279, dec=48.6899232)

    def test_getSpaxel_file_flux_ra_dec_parital(self):
        self._getSpaxel_file_flux_ra_dec(ra=232.5443, dec=48.6899)

    def test_getSpaxel_file_flux_ra_dec_twosigfig(self):
        errMsg = 'some indices are out of limits.'
        self._getSpaxel_file_fail(ra=232.55, dec=48.69, errMsg=errMsg)

    def test_getSpaxel_file_flux_ra_dec_int(self):
        errMsg = 'some indices are out of limits'
        self._getSpaxel_file_fail(ra=232, dec=48, errMsg=errMsg)

    # Tests for getSpaxel from DB
    def _getSpaxel_db_flux_ra_dec(self, ra, dec):
        expect = 0.62007582
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel(cube, 3000, expect, ra=ra, dec=dec)

    def _getSpaxel_db_fail(self, ra, dec, errMsg):
        expect = 0.62007582
        cube = Cube(mangaid=self.mangaid)
        with pytest.raises(MarvinError) as cm:
            self._test_getSpaxel(cube, 3000, expect, ra=ra, dec=dec)
        assert errMsg in str(cm.exception)

    @skipIfNoDB
    def test_getSpaxel_db_flux_ra_dec_full(self):
        self._getSpaxel_db_flux_ra_dec(ra=232.544279, dec=48.6899232)

    @skipIfNoDB
    def test_getSpaxel_db_flux_ra_dec_partial(self):
        self._getSpaxel_db_flux_ra_dec(ra=232.5443, dec=48.6899)

    @skipIfNoDB
    def test_getSpaxel_db_flux_ra_dec_twosigfig(self):
        errMsg = 'some indices are out of limits.'
        self._getSpaxel_db_fail(ra=232.55, dec=48.69, errMsg=errMsg)

    @skipIfNoDB
    def test_getSpaxel_db_flux_ra_dec_int(self):
        errMsg = 'some indices are out of limits.'
        self._getSpaxel_db_fail(ra=232, dec=48, errMsg=errMsg)

    @skipIfNoDB
    def test_getSpaxel_db_flux_x_y(self):
        expect = -0.062497499999999997
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel(cube, 10, expect, x=10, y=5)

    @skipIfNoDB
    def test_getSpaxel_db_flux_x_y_lower(self):
        expect = 0.017929086
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel(cube, 3000, expect, x=10, y=5, xyorig='lower')

    @skipIfNoDB
    def test_getSpaxel_db_flux_x_0_y_0(self):
        expect = 1.0493046
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel(cube, 3000, expect, x=0, y=0)

    @skipIfNoDB
    def test_getSpaxel_db_flux_x_0_y_0_lower(self):
        expect = 0.0
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel(cube, 3000, expect, x=0, y=0, xyorig='lower')

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
