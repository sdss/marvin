#!/usr/bin/env python

import unittest, os
import numpy as np
from marvin.tools.cube import Cube

class TestCube(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.outver = 'v1_5_1'
        cls.filename = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),cls.outver,'8485/stack/manga-8485-1901-LOGCUBE.fits.gz')

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
        self.assertIn('Either filename, mangaid, or plateifu is required!', cm.exception.message)

    def test_cube_loadsuccess(self):
        cube = Cube(filename=self.filename)
        self.assertIsNotNone(cube)
        self.assertEqual(self.filename,cube.filename)

    def _getSpectrum_fail(self, x,y,errMsg):
        cube = Cube(filename=self.filename)
        with self.assertRaises(AssertionError) as cm:
            spectrum = cube.getSpectrum(x,y)
        self.assertIn(errMsg, cm.exception.message)

    def test_getSpectrum_fail_toolargeinput(self):
        self._getSpectrum_fail(1000,1000,'Input x,y coordinates greater than flux dimensions')

    def test_getSpectrum_success(self):
        cube = Cube(filename=self.filename)
        spectrum = cube.getSpectrum(0,0)
        self.assertIsNotNone(spectrum)
        self.assertEqual(np.sum(spectrum),0)
        spectrum = cube.getSpectrum(15,15)
        self.assertIsNotNone(spectrum)
        self.assertNotEqual(np.sum(spectrum),0)

if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
    