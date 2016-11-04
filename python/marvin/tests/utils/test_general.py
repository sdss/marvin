#!/usr/bin/env python
# encoding: utf-8
"""

test_general.py

Created by José Sánchez-Gallego on 7 Apr 2016.
Licensed under a 3-clause BSD license.

Revision history:
    7 Apr 2016 J. Sánchez-Gallego
      Initial version

"""

from __future__ import division
from __future__ import print_function
from marvin.utils.general import convertCoords
from unittest import TestCase
from astropy.io import fits
from astropy.wcs import WCS
import os
import numpy as np
from numpy.testing import assert_allclose
from marvin.tests import TemplateTestCase, Call, template
from marvin.core.exceptions import MarvinError


class TestConvertCoords(TestCase):

    __metaclass__ = TemplateTestCase

    @classmethod
    def setUpClass(cls):
        outver = 'v1_5_1'
        filename = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),
                                outver,
                                '8485/stack/manga-8485-1901-LOGCUBE.fits.gz')
        cls.testHeader = fits.getheader(filename, 1)
        cls.testWcs = WCS(cls.testHeader)
        cls.testShape = fits.getdata(filename, 1).shape[1:]

    def test_pix_center(self):
        """Tests mode='pix', xyorig='center'."""

        coords = [[0, 0],
                  [5, 3],
                  [-5, 1],
                  [1, -5],
                  [10, 10],
                  [-10, -10],
                  [1.5, 2.5],
                  [0.4, 0.25]]

        expected = [[17, 17],
                    [20, 22],
                    [18, 12],
                    [12, 18],
                    [27, 27],
                    [7,  7],
                    [20, 18],
                    [17, 17]]

        cubeCoords = convertCoords(coords, mode='pix', shape=self.testShape)
        assert_allclose(cubeCoords, np.array(expected))

    def test_pix_lower(self):
        """Tests mode='pix', xyorig='lower'."""

        coords = [[0, 0],
                  [5, 3],
                  [10, 10],
                  [1.5, 2.5],
                  [0.4, 0.25]]

        expected = [[0, 0],
                    [3, 5],
                    [10, 10],
                    [2, 2],
                    [0, 0]]

        cubeCoords = convertCoords(coords, mode='pix', shape=self.testShape,
                                   xyorig='lower')
        assert_allclose(cubeCoords, np.array(expected))

    def test_sky(self):
        """Tests mode='sky'."""

        coords = np.array([[232.5447, 48.690201],
                           [232.54259, 48.688948],
                           [232.54135, 48.692415],
                           [232.54285, 48.692372]])

        expected = [[17, 17],
                    [8, 27],
                    [33, 33],
                    [33, 26]]

        cubeCoords = convertCoords(coords, mode='sky', wcs=self.testWcs)

        assert_allclose(cubeCoords, np.array(expected))

    # This allows to do multiple calls to the same test.
    _outside_calls = {
        'pix_center_-50_0': Call(
            {'coords': [[-50, 0]], 'mode': 'pix', 'xyorig': 'center'}, []),
        'pix_center_50_50': Call(
            {'coords': [[50, 50]], 'mode': 'pix', 'xyorig': 'center'}, []),
        'pix_lower_-50_0': Call(
            {'coords': [[-50, 0]], 'mode': 'pix', 'xyorig': 'lower'}, []),
        'pix_center_50_50': Call(
            {'coords': [[50, 50]], 'mode': 'pix', 'xyorig': 'lower'}, []),
        'pix_sky_230_48': Call({'coords': [[230, 48]], 'mode': 'sky'}, []),
        'pix_center_233_48': Call({'coords': [[233, 48]], 'mode': 'sky'}, [])
    }

    @template(_outside_calls)
    def _test_outside(self, kwargs, expected):

        mode = kwargs.get('mode')
        if mode == 'sky':
            kwargs['wcs'] = self.testWcs

        with self.assertRaises(MarvinError) as cm:
            convertCoords(shape=self.testShape, **kwargs)
        self.assertIn('some indices are out of limits', str(cm.exception))
