#!/usr/bin/env python

import os
import unittest

from brain.core.core import URLMapDict
from brain.core.exceptions import BrainError

from marvin import config, marvindb
from marvin.tools.cube import Cube
from marvin.core import MarvinError
from marvin.tests import MarvinTest, skipIfNoDB

import numpy as np
from numpy.testing import assert_allclose

from astropy.io import fits
from astropy import wcs


class TestCubeBase(MarvinTest):

    @classmethod
    def setUpClass(cls):
        cls.mpl = 'MPL-4'
        cls.outver = 'v1_5_1'
        cls.filename = os.path.join(
            os.getenv('MANGA_SPECTRO_REDUX'), cls.outver,
            '8485/stack/manga-8485-1901-LOGCUBE.fits.gz')
        cls.mangaid = '1-209232'
        cls.plate = 8485
        cls.plateifu = '8485-1901'
        cls.cubepk = 10179
        cls.ra = 232.544703894
        cls.dec = 48.6902009334

        cls.init_mode = config.mode
        cls.init_sasurl = config.sasurl
        cls.init_urlmap = config.urlmap
        cls.init_xyorig = config.xyorig

        cls.session = marvindb.session

        cls.cubeFromFile = Cube(filename=cls.filename)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):

        config.sasurl = self.init_sasurl
        config.mode = self.init_mode
        config.urlmap = self.init_urlmap
        config.xyorig = self.init_xyorig

        config.setMPL('MPL-4')

    def tearDown(self):
        if self.session:
            self.session.close()


class TestCube(TestCubeBase):

    def test_mpl_version(self):
        self.assertEqual(config.drpver, self.outver)

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

    def _load_from_db_fail(self, params, errMsg):
        errMsg = 'Could not initialize via db: {0}'.format(errMsg)
        with self.assertRaises(MarvinError) as cm:
            Cube(**params)
        self.assertIn(errMsg, str(cm.exception))

    @skipIfNoDB
    def test_cube_load_from_local_database_nodrpver(self):
        config.drpver = None
        params = {'mangaid': self.mangaid, 'mode': 'local'}
        errMsg = 'drpver not set in config'
        self._load_from_db_fail(params, errMsg)

    @skipIfNoDB
    def test_cube_load_from_local_database_nodbconnected(self):

        # TODO: This tests fails because config.db = None does not disable the
        # local DB, and there is currently no way of doing so.

        config.db = None
        params = {'mangaid': self.mangaid, 'mode': 'local'}
        errMsg = 'No db connected'
        self._load_from_db_fail(params, errMsg)

    @skipIfNoDB
    def test_cube_load_from_local_database_noresultsfound(self):
        params = {'plateifu': '8485-0923', 'mode': 'local'}
        errMsg = 'Could not retrieve cube for plate-ifu {0}: No Results Found'.format(
            params['plateifu'])
        self._load_from_db_fail(params, errMsg)

    @skipIfNoDB
    def test_cube_load_from_local_database_otherexception(self):
        params = {'plateifu': '84.85-1901', 'mode': 'local'}
        errMsg = 'Could not retrieve cube for plate-ifu {0}: Unknown exception'.format(
            params['plateifu'])
        self._load_from_db_fail(params, errMsg)

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

        config.setMPL('MPL-5')
        self.assertEqual(config.drpver, 'v2_0_1')

        cube = Cube(plateifu=self.plateifu, mode='remote', drpver='v1_5_1')
        self.assertEqual(cube._drpver, 'v1_5_1')
        self.assertEqual(cube.hdr['VERSDRP3'].strip(), 'v1_5_0')


class TestGetSpaxel(TestCubeBase):

    #  Tests for getSpaxel
    def _test_getSpaxel(self, cube, idx, expect, **kwargs):
        """Convenience method to test getSpaxel."""

        ext = kwargs.pop('ext', 'flux')
        spectrum = cube.getSpaxel(**kwargs).spectrum
        self.assertAlmostEqual(spectrum[ext][idx], expect, places=5)

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

        expect = -0.10531016
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
        expect = -0.10531016
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

        expect = -0.10531
        self._test_getSpaxel_remote(10, expect, x=10, y=5)

    def test_getSpaxel_remote_ra_dec_success(self):

        expect = 0.62007582
        self._test_getSpaxel_remote(3000, expect, ra=232.544279, dec=48.6899232)

    def _getSpaxel_remote_fail(self, ra, dec, errMsg1, errMsg2, excType=MarvinError):

        cube = Cube(mangaid=self.mangaid, mode='remote')

        with self.assertRaises(excType) as cm:
            cube.getSpaxel(ra=ra, dec=dec)

        self.assertIn(errMsg1, str(cm.exception))
        self.assertIn(errMsg2, str(cm.exception))

    def test_getSpaxel_remote_fail_nourlmap(self):

        self.assertIsNotNone(config.urlmap)
        config.urlmap = URLMapDict()

        with self.assertRaises(BrainError) as cm:
            Cube(mangaid=self.mangaid, mode='remote')

        self.assertIn('No URL Map found', str(cm.exception))
        self.assertIn('Cannot make remote call', str(cm.exception))

    def test_getSpaxel_remote_fail_badresponse(self):

        config.sasurl = 'http://www.averywrongurl.com'
        self.assertIsNotNone(config.urlmap)

        with self.assertRaises(MarvinError) as cm:
            Cube(mangaid=self.mangaid, mode='remote')

        self.assertIn('Failed to establish a new connection', str(cm.exception))

    def test_getSpaxel_remote_fail_badpixcoords(self):

        self.assertIsNotNone(config.urlmap)
        self._getSpaxel_remote_fail(232, 48, 'Something went wrong with the interaction',
                                    'some indices are out of limits.',
                                    excType=BrainError)

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
        expect = -0.10531
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel(cube, 10, expect, x=10, y=5)

    def test_getSpaxel_global_xyorig_lower(self):
        config.xyorig = 'lower'
        expect = 0.017929086
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpaxel(cube, 3000, expect, x=10, y=5)

    def test_getSpaxel_remote_drpver_differ_from_global(self):

        config.setMPL('MPL-5')
        self.assertEqual(config.drpver, 'v2_0_1')

        cube = Cube(plateifu=self.plateifu, mode='remote', drpver='v1_5_1')
        expect = 0.62007582
        self._test_getSpaxel(cube, 3000, expect, ra=232.544279, dec=48.6899232)


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
        self.assertAlmostEqual(cube.wcs.wcs.cd[1, 1], 0.000138889)


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
