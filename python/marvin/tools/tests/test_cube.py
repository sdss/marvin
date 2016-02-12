#!/usr/bin/env python

import os
import unittest
from marvin.tools.cube import Cube


class TestCube(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.outver = 'v1_5_1'
        cls.filename = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'), cls.outver, '8485/stack/manga-8485-1901-LOGCUBE.fits.gz')
        cls.mangaid = '1-209232'

        cls.cubeFromFile = Cube(filename=cls.filename)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cube_loadfail(self):
        with self.assertRaises(AssertionError) as cm:
            Cube()
        self.assertIn('Enter filename, plateifu, or mangaid!',
                      str(cm.exception))

    def test_cube_load_from_local_file_by_filename_success(self):
        cube = Cube(filename=self.filename)
        self.assertIsNotNone(cube)
        self.assertEqual(self.filename, cube.filename)

    def test_cube_load_from_local_file_by_filename_fail(self):
        self.filename = 'not_a_filename.fits'
        self.assertRaises(IOError, lambda: Cube(filename=self.filename))
        # errMsg = '{0} does not exist. Please provide full file path.'.format(self.filename)
        # with self.assertRaises(FileNotFoundError) as cm:
        #     Cube(filename=self.filename)
        # self.assertIn(errMsg, cm.exception.args)

    def test_cube_load_from_local_database_success(self):
        """does not work yet"""
        cube = Cube(mangaid=self.mangaid)
        self.assertIsNotNone(cube)
        self.assertEqual(self.mangaid, cube.mangaid)

    def _test_getSpectrum(self, cube, idx, expect, **kwargs):

        spectrum = cube.getSpectrum(**kwargs)
        self.assertAlmostEqual(spectrum[idx], expect, places=5)

    def _test_getSpectrum_raise_exception(self, message, excType=AssertionError, **kwargs):

        with self.assertRaises(excType) as ee:
            self.cubeFromFile.getSpectrum(**kwargs)

        self.assertIn(message, str(ee.exception))

    def test_getSpectrum_inputs(self):

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

        for xTest, yTest in [(-50, 1), (50, 1), (1, -50), (1, 50)]:
            self._test_getSpectrum_raise_exception(
                'pixel coordinates outside cube', x=xTest, y=yTest)

        for raTest, decTest in [(1., 1.), (100, 60),
                                (232.546383, 1.), (1., 48.6883954)]:
            self._test_getSpectrum_raise_exception(
                'pixel coordinates outside cube', ra=raTest, dec=decTest)

    def test_getSpectrum_file_flux_x_y(self):
        # TODO: check that the expected value is correct.
        expect = -0.10531016
        self._test_getSpectrum(self.cubeFromFile, 10, expect, x=10, y=5)

    def test_getSpectrum_file_flux_ra_dec(self):
        # TODO: check that the expected value is correct.
        expect = 0.017929086
        self._test_getSpectrum(self.cubeFromFile, 3000, expect,
                               ra=232.546383, dec=48.6883954)


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
