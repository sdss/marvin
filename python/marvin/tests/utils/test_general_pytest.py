#!/usr/bin/env python
# encoding: utf-8
"""

test_general_pytest.py

Created by José Sánchez-Gallego on 7 Apr 2016.
Licensed under a 3-clause BSD license.

Revision history:
    7 Apr 2016 J. Sánchez-Gallego
      Initial version
    15 Jun 2016 B. Andrews
      Converted to pytest

"""

from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import os

import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import marvin
from marvin.core.core import DotableCaseInsensitive
from marvin.core.exceptions import MarvinError
from marvin.tests import TemplateTestCase, Call, template
from marvin.utils.general import convertCoords, get_nsa_data, getWCSFromPng, get_plot_params
from marvin.tests import MarvinTest, use_releases
from marvin.utils.dap.datamodel import get_default_plot_params


@pytest.fixture(scope='class')
def testShape(galaxy):
    return fits.getdata(galaxy.cubepath, 1).shape[1:]

@pytest.fixture(scope='class')
def testWcs(galaxy):
    return WCS(fits.getheader(galaxy.cubepath, 1))


class TestConvertCoords(object):

    def test_pix_center(self, testShape):
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
                    [7, 7],
                    [20, 18],
                    [17, 17]]

        cubeCoords = convertCoords(coords, mode='pix', shape=testShape)
        pytest.approx(cubeCoords, np.array(expected))

    def test_pix_lower(self, testShape):
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

        cubeCoords = convertCoords(coords, mode='pix', shape=testShape,
                                   xyorig='lower')
        pytest.approx(cubeCoords, np.array(expected))

    def test_sky(self, testWcs):
        """Tests mode='sky'."""

        coords = np.array([[232.5447, 48.690201],
                           [232.54259, 48.688948],
                           [232.54135, 48.692415],
                           [232.54285, 48.692372]])

        expected = [[17, 17],
                    [8, 27],
                    [33, 33],
                    [33, 26]]

        cubeCoords = convertCoords(coords, mode='sky', wcs=testWcs)

        pytest.approx(cubeCoords, np.array(expected))
    
    @pytest.mark.parametrize('coords, mode, xyorig',
                             [([-50, 0], 'pix', 'center'),
                              ([50, 50], 'pix', 'center'),
                              ([-50, 0], 'pix', 'lower'),
                              ([50, 50], 'pix', 'lower'),
                              ([230, 48], 'sky', None),
                              ([233, 48], 'sky', None)],
                             ids=['-50_0_cen', '50_50_cen', '-50_0_low', '50_50_low',
                                  '230_48_sky', '233_48_sky'])
    def test_coords_outside_cube(self, coords, mode, xyorig, testShape, testWcs):
        kwargs = {'coords': coords, 'mode': mode}

        if xyorig is not None:
            kwargs['xyorig'] = xyorig

        if kwargs['mode'] == 'sky':
            kwargs['wcs'] = testWcs

        with pytest.raises(MarvinError) as cm:
            convertCoords(shape=testShape, **kwargs)

        assert 'some indices are out of limits' in str(cm.value)


class TestGetNSAData(object):

    def _test_nsa(self, data):
        assert isinstance(data, DotableCaseInsensitive)
        assert 'profmean_ivar' in data.keys()
        assert data['profmean_ivar'][0][0] == 18.5536117553711

    def _test_drpall(self, data):
        assert isinstance(data, DotableCaseInsensitive)
        assert 'profmean_ivar' not in data.keys()
        assert 'iauname' in data.keys()
        assert data['iauname'] == 'J153010.73+484124.8'


    def test_nsa(self, galaxy, mode):
        data = get_nsa_data(galaxy.mangaid, source='nsa', mode=mode)
        self._test_nsa(data)

    @use_releases('MPL-5')
    def test_drpall(self, galaxy, mode):
        data = get_nsa_data(galaxy.mangaid, source='drpall', mode=mode)
        self._test_drpall(data)

    def test_auto_nsa_without_db(self, galaxy):
        marvin.config.forceDbOff()
        data = get_nsa_data(galaxy.mangaid, source='nsa', mode='auto')
        self._test_nsa(data)

    @use_releases('MPL-5')
    def test_auto_drpall_without_drpall(self, galaxy):
        marvin.config._drpall = None
        data = get_nsa_data(galaxy.mangaid, source='drpall', mode='auto')
        self._test_drpall(data)

    def test_hybrid_properties_populated(self, galaxy):
        data = get_nsa_data(galaxy.mangaid, source='nsa', mode='local')
        assert 'elpetro_mag_g' in data

    def test_nsa_dotable(self, galaxy):
        data = get_nsa_data(galaxy.mangaid, source='nsa', mode='local')
        assert data['elpetro_mag_g'] == data.elpetro_mag_g

    @use_releases('MPL-5')
    def test_drpall_dotable(self, galaxy):
        data = get_nsa_data(galaxy.mangaid, source='drpall', mode='local')
        assert data['iauname'] == data.iauname

    def test_nsa_old_target_selection(self, release):
        data = get_nsa_data('1-178482', source='nsa', mode='local')
        pytest.approx(data['sersic_flux_ivar'][0], 1.33179831504822)

    def test_nsa_12(self):
        data = get_nsa_data('12-84679', source='nsa', mode='local')
        pytest.approx(data['sersic_flux_ivar'][0], 0.127634227275848)


class TestPillowImage(MarvinTest):

    @classmethod
    def setUpClass(cls):
        super(TestPillowImage, cls).setUpClass()
        outver = 'v1_5_1'
        cls.filename = os.path.join(cls.mangaredux, outver, str(cls.plate), 'stack/images', cls.imgname)

    def test_image_has_wcs(self):
        w = getWCSFromPng(self.filename)
        self.assertEqual(type(w), WCS)

    def test_use_pil(self):
        try:
            import PIL
        except ImportError as e:
            with self.assertRaises(ImportError):
                err = 'No module named PIL'
                self.assertEqual(err, e.args[0])

class TestDataModelPlotParams(MarvinTest):
    
    def bitmasks(self):
        return {'1.1.1': {'badData': {'doNotUse': 0}},
                '2.0.2': {'nocov': 0,
                          'badData': {'unreliable': 5,
                                      'doNotUse': 30}
                          }
                }

    def test_get_plot_params_default_mpl4(self):
        desired = {'bitmasks': self.bitmasks()['1.1.1'],
                   'cmap': 'linearlab',
                   'percentile_clip': [5, 95],
                   'symmetric': False,
                   'snr_min': 1}
        actual = get_plot_params(dapver='1.1.1', prop='emline_gflux')
        self.assertDictEqual(actual, desired)

    def test_get_plot_params_default_mpl5(self):
        desired = {'bitmasks': self.bitmasks()['2.0.2'],
                   'cmap': 'linearlab',
                   'percentile_clip': [5, 95],
                   'symmetric': False,
                   'snr_min': 1}
        actual = get_plot_params(dapver='2.0.2', prop='emline_gflux')
        self.assertDictEqual(actual, desired)

    def test_get_plot_params_vel_mpl4(self):
        desired = {'bitmasks': self.bitmasks()['1.1.1'],
                   'cmap': 'RdBu_r',
                   'percentile_clip': [10, 90],
                   'symmetric': True,
                   'snr_min': None}
        actual = get_plot_params(dapver='1.1.1', prop='stellar_vel')
        self.assertDictEqual(actual, desired)
    
    def test_get_plot_params_vel_mpl5(self):
        desired = {'bitmasks': self.bitmasks()['2.0.2'],
                   'cmap': 'RdBu_r',
                   'percentile_clip': [10, 90],
                   'symmetric': True,
                   'snr_min': None}
        actual = get_plot_params(dapver='2.0.2', prop='stellar_vel')
        self.assertDictEqual(actual, desired)
    
    def test_get_plot_params_sigma_mpl4(self):
        desired = {'bitmasks': self.bitmasks()['1.1.1'],
                   'cmap': 'inferno',
                   'percentile_clip': [10, 90],
                   'symmetric': False,
                   'snr_min': 1}
        actual = get_plot_params(dapver='1.1.1', prop='stellar_sigma')
        self.assertDictEqual(actual, desired)

    def test_get_plot_params_sigma_mpl5(self):
        desired = {'bitmasks': self.bitmasks()['2.0.2'],
                   'cmap': 'inferno',
                   'percentile_clip': [10, 90],
                   'symmetric': False,
                   'snr_min': 1}
        actual = get_plot_params(dapver='2.0.2', prop='stellar_sigma')
        self.assertDictEqual(actual, desired)
