#!/usr/bin/env python
# encoding: utf-8
"""

test_general.py

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

import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import marvin
from marvin.utils.general.structs import DotableCaseInsensitive
from marvin.core.exceptions import MarvinError
from marvin.utils.general import (convertCoords, get_nsa_data, getWCSFromPng, get_mask, get_bits,
                                  get_plot_params)
from marvin.utils.dap.datamodel.base import Bit


mask = np.array([[0, 1], [2, 3]])
bit_lookup = OrderedDict([
    ('BITZERO', Bit(0, 'BITZERO', 'The zeroth bit.')),
    ('BITONE', Bit(1, 'BITONE', 'The first bit.')),
    ('BITTWO', Bit(2, 'BITTWO', 'The second bit.'))
])

@pytest.fixture(scope='function')
def wcs(galaxy):
    return WCS(fits.getheader(galaxy.cubepath, 1))


class TestConvertCoords(object):

    def test_pix_center(self, galaxy):
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

        cubeCoords = convertCoords(coords, mode='pix', shape=galaxy.shape)
        pytest.approx(cubeCoords, np.array(expected))

    def test_pix_lower(self, galaxy):
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

        cubeCoords = convertCoords(coords, mode='pix', shape=galaxy.shape,
                                   xyorig='lower')
        pytest.approx(cubeCoords, np.array(expected))

    def test_sky(self, wcs):
        """Tests mode='sky'."""

        coords = np.array([[232.5447, 48.690201],
                           [232.54259, 48.688948],
                           [232.54135, 48.692415],
                           [232.54285, 48.692372]])

        expected = [[17, 17],
                    [8, 27],
                    [33, 33],
                    [33, 26]]

        cubeCoords = convertCoords(coords, mode='sky', wcs=wcs)

        pytest.approx(cubeCoords, np.array(expected))

    @pytest.mark.parametrize('coords, mode, xyorig',
                             [([-100, 0], 'pix', 'center'),
                              ([100, 100], 'pix', 'center'),
                              ([-100, 0], 'pix', 'lower'),
                              ([100, 100], 'pix', 'lower'),
                              ([230, 48], 'sky', None),
                              ([233, 48], 'sky', None)],
                             ids=['-50_0_cen', '50_50_cen', '-50_0_low', '50_50_low',
                                  '230_48_sky', '233_48_sky'])
    def test_coords_outside_cube(self, coords, mode, xyorig, galaxy, wcs):
        kwargs = {'coords': coords, 'mode': mode}

        if xyorig is not None:
            kwargs['xyorig'] = xyorig

        if kwargs['mode'] == 'sky':
            kwargs['wcs'] = wcs

        with pytest.raises(MarvinError) as cm:
            convertCoords(shape=galaxy.shape, **kwargs)

        assert 'some indices are out of limits' in str(cm.value)


class TestGetNSAData(object):

    def _test_nsa(self, galaxy, data):
        assert isinstance(data, DotableCaseInsensitive)
        assert 'iauname' in data.keys()
        assert data['iauname'] == galaxy.iauname
        assert data['iauname'] == data.iauname
        assert 'profmean_ivar' in data.keys()
        assert 'elpetro_mag_g' in data
        assert 'version' not in data.keys()
        assert data['elpetro_mag_g'] == data.elpetro_mag_g

    def _test_drpall(self, galaxy, data):
        assert isinstance(data, DotableCaseInsensitive)
        assert 'version' in data.keys()
        assert 'profmean_ivar' not in data.keys()
        if galaxy.release != 'MPL-4':
            assert 'iauname' in data.keys()
            assert data['iauname'] == galaxy.iauname
            assert data['iauname'] == data.iauname
            assert 'elpetro_absmag' in data.keys()

    @pytest.mark.parametrize('source', [('nsa'), ('drpall')])
    def test_nsa(self, galaxy, mode, db, source):
        if mode == 'local' and source == 'nsa' and marvin.config.db is None:
            with pytest.raises(MarvinError) as cm:
                data = get_nsa_data(galaxy.mangaid, source=source, mode=mode)
            errmsg = 'get_nsa_data: cannot find a valid DB connection.'
            assert cm.type == MarvinError
            assert errmsg in str(cm.value)
        else:
            data = get_nsa_data(galaxy.mangaid, source=source, mode=mode)
            if source == 'nsa':
                self._test_nsa(galaxy, data)
            elif source == 'drpall':
                self._test_drpall(galaxy, data)

    @pytest.mark.parametrize('monkeyconfig', [('_drpall', None)], indirect=True, ids=['nodrpall'])
    def test_nodrpall(self, galaxy, monkeyconfig):
        data = get_nsa_data(galaxy.mangaid, source='drpall', mode='auto')
        self._test_drpall(galaxy, data)


class TestPillowImage(object):

    def test_image_has_wcs(self, galaxy):
        w = getWCSFromPng(galaxy.imgpath)
        assert isinstance(w, WCS) is True

    def test_use_pil(self):
        try:
            import PIL
        except ImportError as e:
            with pytest.raises(ImportError) as cm:
                err = 'No module named PIL'
                assert err == str(e.args[0])


@pytest.fixture(scope='session')
def bitmask(dapver):
    data = {'1.1.1': {'badData': {'doNotUse': 0}},
            '2.0.2': {'nocov': 0, 'badData': {'unreliable': 5, 'doNotUse': 30}}
            }
    return data[dapver]


class TestMaskbit(object):

    def test_get_mask_no_bitnames(self, mask=mask, bit_lookup=bit_lookup):
        actual = get_mask(mask, bit_lookup, bitnames=())
        expected = mask > 0
        assert (actual == expected).all()

    def test_get_mask_one_bitname_as_string(self, mask=mask, bit_lookup=bit_lookup):
        actual = get_mask(mask, bit_lookup, bitnames='BITZERO')
        expected = (mask & 2**0) > 0
        assert (actual == expected).all()

    def test_get_mask_one_bitname_as_string_lower(self, mask=mask, bit_lookup=bit_lookup):
        actual = get_mask(mask, bit_lookup, bitnames='bitzero')
        expected = (mask & 2**0) > 0
        assert (actual == expected).all()

    def test_get_mask_one_bitname_as_list(self, mask=mask, bit_lookup=bit_lookup):
        actual = get_mask(mask, bit_lookup, bitnames=('BITZERO',))
        expected = (mask & 2**0) > 0
        assert (actual == expected).all()
    
    def test_get_mask_multiple_bitnames(self, mask=mask, bit_lookup=bit_lookup):
        actual = get_mask(mask, bit_lookup, bitnames=('BITZERO', 'BITONE'))
        mask0 = (mask & 2**0) > 0
        mask1 = (mask & 2**1) > 0
        expected = np.logical_or.reduce((mask0, mask1))
        assert (actual == expected).all()

    def test_get_mask_invalid_bitnames(self, mask=mask, bit_lookup=bit_lookup):
        with pytest.raises(MarvinError) as cm:
            get_mask(mask, bit_lookup, bitnames=('NOT_A_BITNAME',))

        assert 'Invalid mask bit name:' in str(cm)

    def test_get_bits_name(self, bit_lookup=bit_lookup):
        actual = get_bits(3, bit_lookup, output='name')
        expected = ['BITZERO', 'BITONE']
        assert actual == expected

    def test_get_bits_value(self, bit_lookup=bit_lookup):
        actual = get_bits(3, bit_lookup, output='value')
        expected = [0, 1]
        assert actual == expected

    def test_get_bits_object(self, bit_lookup=bit_lookup):
        actual = get_bits(3, bit_lookup, output='object')
        expected = [bit_lookup['BITZERO'], bit_lookup['BITONE']]
        assert actual == expected

    def test_get_bits_zero(self, bit_lookup=bit_lookup):
        actual = get_bits(0, bit_lookup, output='object')
        expected = []
        assert actual == expected


class TestDataModelPlotParams(object):

    @pytest.mark.parametrize('name, desired',
                             [('emline_gflux', {'cmap': 'linearlab', 'percentile_clip': [5, 95], 'symmetric': False, 'snr_min': 1}),
                              ('stellar_vel', {'cmap': 'RdBu_r', 'percentile_clip': [10, 90], 'symmetric': True, 'snr_min': None}),
                              ('stellar_sigma', {'cmap': 'inferno', 'percentile_clip': [10, 90], 'symmetric': False, 'snr_min': 1})],
                             ids=['emline', 'stvel', 'stsig'])
    def test_get_plot_params(self, bitmask, dapver, name, desired):
        desired['bitmasks'] = bitmask
        actual = get_plot_params(dapver=dapver, prop=name)
        assert desired == actual
