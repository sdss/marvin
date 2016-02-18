#!/usr/bin/env python

import os
import copy
import unittest
from marvin.tools.cube import Cube
from marvin.tools.core import MarvinError
from marvin import config, session, datadb
from marvin.tools.tests import MarvinTest, skipIfNoDB


class TestCube(MarvinTest):

    @classmethod
    def setUpClass(cls):
        cls.outver = 'v1_5_1'
        cls.filename = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'), cls.outver, '8485/stack/manga-8485-1901-LOGCUBE.fits.gz')
        cls.mangaid = '1-209232'
        cls.plate = 8485
        cls.plateifu = '8485-1901'
        cls.cubepk = 10179
        cls.ra = 232.544703894
        cls.dec = 48.6902009334
        cls.initconfig = copy.deepcopy(config)
        cls.session = session

        cls.cubeFromFile = Cube(filename=cls.filename)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # reset config variables
        cvars = ['mode', 'drpver', 'dapver', 'db']
        for var in cvars:
            config.__setattr__(var, self.initconfig.__getattribute__(var))
        config.drpver = self.outver
        session = self.session

    def tearDown(self):
        if self.session:
            self.session.close()

    # Tests for Cube Load by File
    def test_cube_loadfail(self):
        with self.assertRaises(AssertionError) as cm:
            cube = Cube()
        self.assertIn('Enter filename, plateifu, or mangaid!', str(cm.exception))

    def test_cube_load_from_local_file_by_filename_success(self):
        cube = Cube(filename=self.filename)
        self.assertIsNotNone(cube)
        self.assertEqual(self.filename, cube.filename)

    def test_cube_load_from_local_file_by_filename_fail(self):
        self.filename = 'not_a_filename.fits'
        self.assertRaises(MarvinError, lambda: Cube(filename=self.filename))
        # errMsg = '{0} does not exist. Please provide full file path.'.format(self.filename)
        # with self.assertRaises(FileNotFoundError) as cm:
        #     Cube(filename=self.filename)
        # self.assertIn(errMsg, cm.exception.args)

    # Tests for Cube Load by Database
    @skipIfNoDB
    def test_cube_load_from_local_database_success(self):
        cube = Cube(mangaid=self.mangaid)
        self.assertIsNotNone(cube)
        self.assertEqual(self.mangaid, cube.mangaid)
        self.assertEqual(self.plate, cube.plate)
        self.assertEqual(self.dec, cube.dec)
        self.assertEqual(self.ra, cube.ra)

    def _load_from_db_fail(self, params, errMsg):
        errMsg = 'Could not initialize via db: {0}'.format(errMsg)
        with self.assertRaises(MarvinError) as cm:
            cube = Cube(**params)
        self.assertIn(errMsg, str(cm.exception))

    @skipIfNoDB
    def test_cube_load_from_local_database_nodrpver(self):
        config.drpver = None
        params = {'mangaid': self.mangaid}
        errMsg = 'drpver not set in config'
        self._load_from_db_fail(params, errMsg)

    @skipIfNoDB
    def test_cube_load_from_local_database_nodbconnected(self):
        config.db = None
        params = {'mangaid': self.mangaid}
        errMsg = 'No db connected'
        self._load_from_db_fail(params, errMsg)

    @skipIfNoDB
    def test_cube_load_from_local_database_noresultsfound(self):
        params = {'plateifu': '8485-0923'}
        errMsg = 'Could not retrieve cube for plate-ifu {0}: No Results Found'.format(params['plateifu'])
        self._load_from_db_fail(params, errMsg)

    @skipIfNoDB
    def test_cube_load_from_local_database_otherexception(self):
        params = {'plateifu': '84.85-1901'}
        errMsg = 'Could not retrieve cube for plate-ifu {0}: Unknown exception'.format(params['plateifu'])
        self._load_from_db_fail(params, errMsg)

    @skipIfNoDB
    def test_cube_load_from_local_database_multipleresultsfound(self):
        params = {'plateifu': self.plateifu}
        errMsg = 'Could not retrieve cube for plate-ifu {0}: Multiple Results Found'.format(params['plateifu'])
        newrow = {'plate': '8485', 'mangaid': self.mangaid, 'ifudesign_pk': 12, 'pipeline_info_pk': 21}
        self._addToDB(datadb.Cube, newrow)
        self._load_from_db_fail(params, errMsg)

    def _addToDB(self, table, colvaldict):
        self.session.begin()
        param = table()
        for column, value in colvaldict.iteritems():
            param.__setattr__(column, value)
        self.session.add(param)
        self.session.flush()

    #  Tests for getSpectrum
    def _test_getSpectrum(self, cube, idx, expect, **kwargs):
        """Convenience method to test getSpectrum."""

        spectrum = cube.getSpectrum(**kwargs)
        self.assertAlmostEqual(spectrum[idx], expect, places=5)

    def _test_getSpectrum_raise_exception(self, message, excType=AssertionError, **kwargs):
        """Convenience method to test exceptions raised by getSpectrum."""

        with self.assertRaises(excType) as ee:
            self.cubeFromFile.getSpectrum(**kwargs)

        self.assertIn(message, str(ee.exception))

    def test_getSpectrum_inputs(self):
        """Tests exceptions when getSpectrum gets inappropriate inputs."""

        self._test_getSpectrum_raise_exception(
            'Either use (x, y) or (ra, dec)', x=1, ra=1)

        self._test_getSpectrum_raise_exception(
            'Either use (x, y) or (ra, dec)', x=1, dec=1, ra=1)

        self._test_getSpectrum_raise_exception('Specify both x and y', x=1)

        self._test_getSpectrum_raise_exception('Specify both ra and dec', ra=1)

        self._test_getSpectrum_raise_exception(
            'You need to specify either (x, y) or (ra, dec)',
            excType=ValueError)

    def test_getSpectrum_outside_cube(self):
        """Tests getSpectrum when the input coords are outside the cube."""

        for xTest, yTest in [(-50, 1), (50, 1), (1, -50), (1, 50)]:
            self._test_getSpectrum_raise_exception(
                'pixel coordinates outside cube', x=xTest, y=yTest)

        for raTest, decTest in [(1., 1.), (100, 60),
                                (232.546383, 1.), (1., 48.6883954)]:
            self._test_getSpectrum_raise_exception(
                'pixel coordinates outside cube', ra=raTest, dec=decTest)

    # Tests for getSpectrum from file
    def test_getSpectrum_file_flux_x_y(self):
        """Tests getSpectrum from a file cube with x, y inputs."""
        # TODO: check that the expected value is correct.

        expect = -0.10531016
        self._test_getSpectrum(self.cubeFromFile, 10, expect, x=10, y=5)

    def _getSpectrum_file_flux_ra_dec(self, ra, dec):
        """Tests getSpectrum from a file cube with ra, dec inputs."""
        # TODO: check that the expected value is correct.
        expect = 0.017929086
        self._test_getSpectrum(self.cubeFromFile, 3000, expect, ra=ra, dec=dec)

    def _getSpectrum_file_fail(self, ra, dec, errMsg):
        expect = 0.017929086
        with self.assertRaises(AssertionError) as cm:
            self._test_getSpectrum(self.cubeFromFile, 3000, expect, ra=ra, dec=dec)
        self.assertIn(errMsg, str(cm.exception))

    def test_getSpectrum_file_flux_ra_dec_full(self):
        self._getSpectrum_file_flux_ra_dec(ra=232.546383, dec=48.6883954)

    def test_getSpectrum_file_flux_ra_dec_parital(self):
        self._getSpectrum_file_flux_ra_dec(ra=232.5464, dec=48.6884)

    def test_getSpectrum_file_flux_ra_dec_twosigfig(self):
        errMsg = 'pixel coordinates outside cube'
        self._getSpectrum_file_fail(ra=232.55, dec=48.69, errMsg=errMsg)

    def test_getSpectrum_file_flux_ra_dec_int(self):
        errMsg = 'pixel coordinates outside cube'
        self._getSpectrum_file_fail(ra=232, dec=48, errMsg=errMsg)

    # Tests for getSpectrum from DB
    def _getSpectrum_db_flux_ra_dec(self, ra, dec):
        expect = 0.017929086
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpectrum(cube, 3000, expect, ra=ra, dec=dec)

    def _getSpectrum_db_fail(self, ra, dec, errMsg):
        expect = 0.017929086
        cube = Cube(mangaid=self.mangaid)
        with self.assertRaises(AssertionError) as cm:
            self._test_getSpectrum(cube, 3000, expect, ra=ra, dec=dec)
        self.assertIn(errMsg, str(cm.exception))

    @skipIfNoDB
    def test_getSpectrum_db_flux_ra_dec_full(self):
        self._getSpectrum_db_flux_ra_dec(ra=232.546383, dec=48.6883954)

    @skipIfNoDB
    def test_getSpectrum_db_flux_ra_dec_partial(self):
        self._getSpectrum_db_flux_ra_dec(ra=232.5464, dec=48.6884)

    @skipIfNoDB
    def test_getSpectrum_db_flux_ra_dec_twosigfig(self):
        errMsg = 'pixel coordinates outside cube'
        self._getSpectrum_db_fail(ra=232.55, dec=48.69, errMsg=errMsg)

    @skipIfNoDB
    def test_getSpectrum_db_flux_ra_dec_int(self):
        errMsg = 'pixel coordinates outside cube'
        self._getSpectrum_db_fail(ra=232, dec=48, errMsg=errMsg)

    @skipIfNoDB
    def test_getSpectrum_db_flux_x_y(self):
        expect = -0.10531016
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpectrum(cube, 10, expect, x=10, y=5)

    # Tests for getSpectrum remotely
    def _getSpectrum_remote(self, **kwargs):
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpectrum(cube, 10, expect, x=10, y=5)

    def test_getSpectrum_remote_x_y_success(self):
        config.sasurl = 'http://5aafb8e.ngrok.com'
        config.mode = 'remote'
        expect = -0.10531016
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpectrum(cube, 10, expect, x=10, y=5)

    def test_getSpectrum_remote_ra_dec_success(self):
        config.sasurl = 'http://5aafb8e.ngrok.com'
        config.mode = 'remote'
        expect = 0.017929086
        cube = Cube(mangaid=self.mangaid)
        self._test_getSpectrum(cube, 3000, expect, ra=232.546383, dec=48.6883954)

    def _getSpectrum_remote_fail(self, ra, dec, errMsg1, errMsg2):
        cube = Cube(mangaid=self.mangaid)
        with self.assertRaises(MarvinError) as cm:
            flux = cube.getSpectrum(ra=ra, dec=dec)
        self.assertIn(errMsg1, str(cm.exception))
        self.assertIn(errMsg2, str(cm.exception))

    def test_getSpectrum_remote_fail_badresponse(self):
        config.sasurl = 'http://wrong.url.com'
        config.mode = 'remote'
        self._getSpectrum_remote_fail(self.ra, self.dec, 'Error retrieving response', 'Http status code 404')

    def test_getSpectrum_remote_fail_badpixcoords(self):
        config.sasurl = 'http://5aafb8e.ngrok.com'
        config.mode = 'remote'
        self._getSpectrum_remote_fail(232, 48, 'Could not retrieve spaxels remotely', 'pixel coordinates outside cube')

if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
