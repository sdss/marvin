#!/usr/bin/env python

import os
import unittest
import numpy as np
from marvin.tools.cube import Cube


class TestCube(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.outver = 'v1_5_1'
        cls.filename = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'), cls.outver, '8485/stack/manga-8485-1901-LOGCUBE.fits.gz')

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cube_loadfail(self):
        with self.assertRaises(AssertionError) as cm:
            cube = Cube()
        self.assertIn('Enter filename, plateifu, or mangaid!', str(cm.exception))

    def test_local_cube_load_by_filename_success(self):
        cube = Cube(filename=self.filename)
        self.assertIsNotNone(cube)
        self.assertEqual(self.filename, cube.filename)

    def test_local_cube_load_by_filename_fail(self):
        self.filename = 'not_a_filename.fits'
        # errMsg = Exception('{0} does not exist. Please provide full file path.'.format(self.filename))
        with self.assertRaises(AssertionError) as cm:
            cube = Cube(filename=self.filename)
        print('here')
        print(cm.exception)
        self.assertEqual(FileNotFoundError, cm.exception)




    # def _getSpectrum_fail(self, x, y, errMsg):
    #     cube = Cube(filename=self.filename)
    #     with self.assertRaises(AssertionError) as cm:
    #         spectrum = cube.getSpectrum(x, y)
    #     self.assertIn(errMsg, str(cm.exception))

    # def test_getSpectrum_fail_toolargeinput(self):
    #     self._getSpectrum_fail(100000, 100000, 'Input x,y coordinates greater than flux dimensions')

    # def test_getSpectrum_success(self):
    #     cube = Cube(filename=self.filename)
    #     spectrum = cube.getSpectrum(0, 0)
    #     self.assertIsNotNone(spectrum)
    #     self.assertEqual(np.sum(spectrum), 0)
    #     spectrum = cube.getSpectrum(15, 15)
    #     self.assertIsNotNone(spectrum)
    #     self.assertNotEqual(np.sum(spectrum), 0)

if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
