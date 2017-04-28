#!/usr/bin/env python

import os
import unittest

from astropy.io import fits

from marvin import config, marvindb
from marvin.tools.cube import Cube
from marvin.core.core import DotableCaseInsensitive
from marvin.core.exceptions import MarvinError
from marvin.tests import MarvinTest, skipIfNoDB

import numpy as np
from numpy.testing import assert_allclose

from astropy import wcs


class TestCubeBase(MarvinTest):

    @classmethod
    def setUpClass(cls):

        super(TestCubeBase, cls).setUpClass()

        cls.outrelease = 'MPL-4'
        cls._update_release(cls.outrelease)
        cls.set_filepaths()
        cls.filename = os.path.realpath(cls.cubepath)
        cls.cubeFromFile = Cube(filename=cls.filename)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self._reset_the_config()
        self.set_sasurl('local')
        self._update_release('MPL-4')
        self.set_filepaths()

    def tearDown(self):
        if self.session:
            self.session.close()


class TestCube(TestCubeBase):

    def test_mpl_version(self):
        self.assertEqual(config.release, self.outrelease)

    # Tests for Cube Load by File
    def test_cube_loadfail(self):
        with self.assertRaises(AssertionError) as cm:
            Cube()
        self.assertIn('Enter filename, plateifu, or mangaid!', str(cm.exception))

    def test_cube_load_from_local_file_by_filename_success(self):
        cube = Cube(filename=self.filename)
        self.assertIsNotNone(cube)
        self.assertEqual(self.filename, cube.filename)

    def test_cube_load_from_local_file_by_filename_fail(self):
        self.filename = 'not_a_filename.fits'
        self.assertRaises(MarvinError, lambda: Cube(filename=self.filename))

    @skipIfNoDB
    def test_cube_load_from_local_database_success(self):
        """Tests for Cube Load by Database."""

        cube = Cube(mangaid=self.mangaid)
        self.assertIsNotNone(cube)
        self.assertEqual(self.mangaid, cube.mangaid)
        self.assertEqual(self.plate, cube.plate)
        self.assertEqual(self.dec, cube.dec)
        self.assertEqual(self.ra, cube.ra)

    def _load_from_db_fail(self, params, errMsg, errType=MarvinError):
        with self.assertRaises(errType) as cm:
            Cube(**params)
        self.assertIn(errMsg, str(cm.exception))

    @skipIfNoDB
    @unittest.expectedFailure
    def test_cube_load_from_local_database_nodbconnected(self):

        # TODO: This tests fails because config.db = None does not disable the
        # local DB, and there is currently no way of doing so.

        # need to undo setting the config.db to None so that subsequent tests will pass
        # config.db = None
        params = {'mangaid': self.mangaid, 'mode': 'local'}
        errMsg = 'No db connected'
        self._load_from_db_fail(params, errMsg)

    @skipIfNoDB
    def test_cube_load_from_local_database_noresultsfound(self):
        params = {'plateifu': '8485-0923', 'mode': 'local'}
        errMsg = 'Could not retrieve cube for plate-ifu {0}: No Results Found'.format(
            params['plateifu'])
        self._load_from_db_fail(params, errMsg, errType=MarvinError)

    @skipIfNoDB
    def test_cube_load_from_local_database_otherexception(self):
        params = {'plateifu': '84.85-1901', 'mode': 'local'}
        errMsg = 'Could not retrieve cube for plate-ifu {0}: Unknown exception'.format(
            params['plateifu'])
        self._load_from_db_fail(params, errMsg, errType=MarvinError)

    # @skipIfNoDB
    # def test_cube_load_from_local_database_multipleresultsfound(self):
    #     params = {'plateifu': self.plateifu, 'mode': 'local'}
    #     errMsg = 'Could not retrieve cube for plate-ifu {0}: Multiple Results Found'.format(
    #         params['plateifu'])
    #     newrow = {'plate': '8485', 'mangaid': self.mangaid,
    #               'ifudesign_pk': 12, 'pipeline_info_pk': 21}
    #     self._addToDB(marvindb.datadb.Cube, newrow)
    #     self._load_from_db_fail(params, errMsg)
    #
    # def _addToDB(self, table, colvaldict):
    #     self.session.begin()
    #     param = table()
    #     for column, value in colvaldict.iteritems():
    #         param.__setattr__(column, value)
    #     self.session.add(param)
    #     self.session.flush()

    def test_cube_flux_from_local_database(self):

        cube = Cube(plateifu=self.plateifu, mode='local')
        flux = cube.flux
        self.assertEqual(cube.data_origin, 'db')

        cubeFlux = fits.getdata(self.filename)

        self.assertTrue(np.allclose(flux, cubeFlux))

    def test_cube_remote_drpver_differ_from_global(self):

        # This tests requires having the cube for 8485-1901 loaded for both
        # MPL-4 and MPL-5.

        self._update_release('MPL-5')
        self.assertEqual(config.release, 'MPL-5')

        cube = Cube(plateifu=self.plateifu, mode='remote', release='MPL-4')
        self.assertEqual(cube._drpver, 'v1_5_1')
        self.assertEqual(cube.header['VERSDRP3'].strip(), 'v1_5_0')

    def test_cube_file_redshift(self):
        cube = Cube(filename=self.filename)
        self.assertAlmostEqual(cube.nsa.redshift, 0.0407447)

    def test_cube_db_redshift(self):
        cube = Cube(plateifu=self.plateifu, mode='local')
        self.assertAlmostEqual(cube.nsa.z, 0.0407447)

    def test_cube_remote_redshift(self):
        cube = Cube(plateifu=self.plateifu, mode='remote')
        self.assertAlmostEqual(cube.nsa.z, 0.0407447)

    def _test_nsa(self, nsa_data, mode='nsa'):
        self.assertIsInstance(nsa_data, DotableCaseInsensitive)
        if mode == 'drpall':
            self.assertNotIn('profmean_ivar', nsa_data.keys())
        self.assertIn('zdist', nsa_data.keys())
        self.assertAlmostEqual(nsa_data['zdist'], 0.041201399999999999)

    def test_nsa_file_auto(self):
        cube = Cube(filename=self.filename)
        self.assertEqual(cube.nsa_source, 'auto')
        self._test_nsa(cube.nsa)

    def test_nsa_file_nsa(self):
        cube = Cube(filename=self.filename, nsa_source='nsa')
        self.assertEqual(cube.nsa_source, 'nsa')
        self._test_nsa(cube.nsa)

    def test_nsa_file_drpall(self):
        cube = Cube(plateifu=self.plateifu, nsa_source='drpall')
        self.assertEqual(cube.nsa_source, 'drpall')
        self._test_nsa(cube.nsa)

    def test_nsa_db_auto(self):
        # import pdb; pdb.set_trace()
        cube = Cube(plateifu=self.plateifu)
        self.assertEqual(cube.nsa_source, 'auto')
        self._test_nsa(cube.nsa)

    def test_nsa_db_nsa(self):
        cube = Cube(plateifu=self.plateifu, nsa_source='nsa')
        self.assertEqual(cube.nsa_source, 'nsa')
        self._test_nsa(cube.nsa)

    def test_nsa_db_drpall(self):
        cube = Cube(plateifu=self.plateifu, nsa_source='drpall')
        self.assertEqual(cube.nsa_source, 'drpall')
        self._test_nsa(cube.nsa)

    def test_nsa_remote_auto(self):
        cube = Cube(plateifu=self.plateifu, mode='remote')
        self.assertEqual(cube.nsa_source, 'auto')
        self._test_nsa(cube.nsa)

    def test_nsa_remote_nsa(self):
        cube = Cube(plateifu=self.plateifu, mode='remote', nsa_source='nsa')
        self.assertEqual(cube.nsa_source, 'nsa')
        self._test_nsa(cube.nsa)

    def test_nsa_remote_drpall(self):
        cube = Cube(plateifu=self.plateifu, mode='remote', nsa_source='drpall')
        self.assertEqual(cube.nsa_source, 'drpall')
        self._test_nsa(cube.nsa)

    def test_release(self):
        cube = Cube(plateifu=self.plateifu)
        self.assertEqual(cube.release, 'MPL-4')

    def test_set_release_fails(self):
        cube = Cube(plateifu=self.plateifu)
        with self.assertRaises(MarvinError) as ee:
            cube.release = 'a'
            self.assertIn('the release cannot be changed', str(ee.exception))

    def test_load_7443_12701_file(self):
        """Loads a cube that is not in the NSA catalogue."""

        self._update_release('MPL-5')
        self.set_filepaths()
        filename = os.path.realpath(os.path.join(
            self.drppath, '7443/stack/manga-7443-12701-LOGCUBE.fits.gz'))
        cube = Cube(filename=filename)
        self.assertEqual(cube.data_origin, 'file')
        self.assertIn('elpetro_amivar', cube.nsa)

    def test_load_7443_12701_db(self):
        """Loads a cube that is not in the NSA catalogue."""

        self._update_release('MPL-5')
        cube = Cube(plateifu='7443-12701')
        self.assertEqual(cube.data_origin, 'db')
        self.assertIsNone(cube.nsa)

    def test_load_7443_12701_api(self):
        """Loads a cube that is not in the NSA catalogue."""

        self._update_release('MPL-5')
        cube = Cube(plateifu='7443-12701', mode='remote')
        self.assertEqual(cube.data_origin, 'api')
        self.assertIsNone(cube.nsa)


class TestGetSpaxel(TestCubeBase):

    #  Tests for getSpaxel
    def _test_getSpaxel(self, cube, idx, expect, **kwargs):
        """Convenience method to test getSpaxel."""

        ext = kwargs.pop('ext', 'flux')
        spectrum = cube.getSpaxel(**kwargs).spectrum
        self.assertAlmostEqual(getattr(spectrum, ext)[idx], expect, places=5)

    def _test_getSpaxel_raise_exception(self, message, excType=AssertionError, **kwargs):
        """Convenience method to test exceptions raised by getSpaxel."""

        with self.assertRaises(excType) as ee:
            self.cubeFromFile.getSpaxel(**kwargs)

        self.assertIn(message, str(ee.exception))

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
        with self.assertRaises(MarvinError) as cm:
            self._test_getSpaxel(self.cubeFromFile, 3000, expect, ra=ra, dec=dec)
        self.assertIn(errMsg, str(cm.exception))

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
        with self.assertRaises(MarvinError) as cm:
            self._test_getSpaxel(cube, 3000, expect, ra=ra, dec=dec)
        self.assertIn(errMsg, str(cm.exception))

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

        with self.assertRaises(excType) as cm:
            cube.getSpaxel(ra=ra, dec=dec)

        self.assertIn(errMsg1, str(cm.exception))
        if errMsg2:
            self.assertIn(errMsg2, str(cm.exception))

    def test_getSpaxel_remote_fail_badresponse(self):

        config.sasurl = 'http://www.averywrongurl.com'
        self.assertIsNotNone(config.urlmap)

        with self.assertRaises(MarvinError) as cm:
            Cube(mangaid=self.mangaid, mode='remote')

        self.assertIn('Failed to establish a new connection', str(cm.exception))

    def test_getSpaxel_remote_fail_badpixcoords(self):

        self.assertIsNotNone(config.urlmap)
        self._getSpaxel_remote_fail(232, 48, 'some indices are out of limits.')

    def _test_getSpaxel_array(self, cube, nCoords, specIndex, expected, **kwargs):
        """Tests getSpaxel with array coordinates."""

        spaxels = cube.getSpaxel(**kwargs)

        self.assertEqual(len(spaxels), nCoords)
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
        self.assertEqual(config.release, 'MPL-5')

        cube = Cube(plateifu=self.plateifu, mode='remote', release='MPL-4')
        expect = 0.62007582
        self._test_getSpaxel(cube, 3000, expect, ra=232.544279, dec=48.6899232)

    @skipIfNoDB
    def test_getspaxel_matches_file_db_remote(self):

        self._update_release('MPL-4')
        self.assertEqual(config.release, 'MPL-4')

        cube_file = Cube(filename=self.filename)
        cube_db = Cube(plateifu=self.plateifu)
        cube_api = Cube(plateifu=self.plateifu, mode='remote')

        self.assertEqual(cube_file.data_origin, 'file')
        self.assertEqual(cube_db.data_origin, 'db')
        self.assertEqual(cube_api.data_origin, 'api')

        xx = 12
        yy = 5
        spec_idx = 200

        spaxel_slice_file = cube_file[xx, yy]
        spaxel_slice_db = cube_db[xx, yy]
        spaxel_slice_api = cube_api[xx, yy]

        flux_result = 0.017639931
        ivar_result = 352.12421
        mask_result = 1026

        self.assertAlmostEqual(spaxel_slice_file.spectrum.flux[spec_idx], flux_result)
        self.assertAlmostEqual(spaxel_slice_db.spectrum.flux[spec_idx], flux_result)
        self.assertAlmostEqual(spaxel_slice_api.spectrum.flux[spec_idx], flux_result)

        self.assertAlmostEqual(spaxel_slice_file.spectrum.ivar[spec_idx], ivar_result, places=5)
        self.assertAlmostEqual(spaxel_slice_db.spectrum.ivar[spec_idx], ivar_result, places=3)
        self.assertAlmostEqual(spaxel_slice_api.spectrum.ivar[spec_idx], ivar_result, places=3)

        self.assertAlmostEqual(spaxel_slice_file.spectrum.mask[spec_idx], mask_result)
        self.assertAlmostEqual(spaxel_slice_db.spectrum.mask[spec_idx], mask_result)
        self.assertAlmostEqual(spaxel_slice_api.spectrum.mask[spec_idx], mask_result)

        xx_cen = -5
        yy_cen = -12

        spaxel_getspaxel_file = cube_file.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_db = cube_db.getSpaxel(x=xx_cen, y=yy_cen)
        spaxel_getspaxel_api = cube_api.getSpaxel(x=xx_cen, y=yy_cen)

        self.assertAlmostEqual(spaxel_getspaxel_file.spectrum.flux[spec_idx], flux_result)
        self.assertAlmostEqual(spaxel_getspaxel_db.spectrum.flux[spec_idx], flux_result)
        self.assertAlmostEqual(spaxel_getspaxel_api.spectrum.flux[spec_idx], flux_result)

        self.assertAlmostEqual(spaxel_getspaxel_file.spectrum.ivar[spec_idx],
                               ivar_result, places=5)
        self.assertAlmostEqual(spaxel_getspaxel_db.spectrum.ivar[spec_idx],
                               ivar_result, places=3)
        self.assertAlmostEqual(spaxel_getspaxel_api.spectrum.ivar[spec_idx],
                               ivar_result, places=3)

        self.assertAlmostEqual(spaxel_getspaxel_file.spectrum.mask[spec_idx], mask_result)
        self.assertAlmostEqual(spaxel_getspaxel_db.spectrum.mask[spec_idx], mask_result)
        self.assertAlmostEqual(spaxel_getspaxel_api.spectrum.mask[spec_idx], mask_result)


class TestWCS(TestCubeBase):

    def test_wcs_file(self):
        cube = Cube(filename=self.filename)
        self.assertIsInstance(cube.wcs, wcs.WCS)
        self.assertAlmostEqual(cube.wcs.wcs.cd[1, 1], 0.000138889)

    def test_wcs_db(self):
        cube = Cube(plateifu=self.plateifu)
        self.assertIsInstance(cube.wcs, wcs.WCS)
        self.assertAlmostEqual(cube.wcs.wcs.cd[1, 1], 0.000138889)

    def test_wcs_api(self):
        cube = Cube(plateifu=self.plateifu, mode='remote')
        self.assertIsInstance(cube.wcs, wcs.WCS)
        self.assertAlmostEqual(cube.wcs.wcs.pc[1, 1], 0.000138889)


class TestPickling(TestCubeBase):

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
        self.assertEqual(cube.data_origin, 'file')
        self.assertIsInstance(cube, Cube)
        self.assertIsNotNone(cube.data)

        path = cube.save()
        self._files_created.append(path)

        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.realpath(path),
                         os.path.realpath(self.filename[0:-7] + 'mpf'))
        self.assertIsNotNone(cube.data)

        cube = None
        self.assertIsNone(cube)

        cube_restored = Cube.restore(path)
        self.assertEqual(cube_restored.data_origin, 'file')
        self.assertIsInstance(cube_restored, Cube)
        self.assertIsNotNone(cube_restored.data)

    def test_pickling_file_custom_path(self):

        cube = Cube(filename=self.filename)

        test_path = '~/test.mpf'
        path = cube.save(path=test_path)
        self._files_created.append(path)

        self.assertTrue(os.path.exists(path))
        self.assertEqual(path, os.path.realpath(os.path.expanduser(test_path)))

        cube_restored = Cube.restore(path, delete=True)
        self.assertEqual(cube_restored.data_origin, 'file')
        self.assertIsInstance(cube_restored, Cube)
        self.assertIsNotNone(cube_restored.data)

        self.assertFalse(os.path.exists(path))

    def test_pickling_db(self):

        cube = Cube(plateifu=self.plateifu)
        self.assertEqual(cube.data_origin, 'db')

        with self.assertRaises(MarvinError) as ee:
            cube.save()

        self.assertIn('objects with data_origin=\'db\' cannot be saved.',
                      str(ee.exception))

    def test_pickling_api(self):

        cube = Cube(plateifu=self.plateifu, mode='remote')
        self.assertEqual(cube.data_origin, 'api')
        self.assertIsInstance(cube, Cube)
        self.assertIsNone(cube.data)

        path = cube.save()
        self._files_created.append(path)

        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.realpath(path),
                         os.path.realpath(self.filename[0:-7] + 'mpf'))

        cube = None
        self.assertIsNone(cube)

        cube_restored = Cube.restore(path)
        self.assertEqual(cube_restored.data_origin, 'api')
        self.assertIsInstance(cube_restored, Cube)
        self.assertIsNone(cube_restored.data)
        self.assertEqual(cube_restored.header['VERSDRP3'], 'v1_5_0')


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
