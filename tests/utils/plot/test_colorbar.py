#!/usr/bin/env python
# encoding: utf-8
#
# test_colorbar.py
#
# @Author: Brett Andrews <andrews>
# @Date:   2017-07-26 11:07:00
# @Last modified by:   andrews
# @Last modified time: 2017-11-27 12:11:94

import os

import pytest
import numpy as np

from marvin.utils.plot import colorbar


@pytest.fixture(scope='function')
def image():
    data = np.linspace(1, 9, 9).reshape(3, 3)
    mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    masked_array = np.ma.array(data, mask=mask)
    return masked_array


class TestColorMap(object):

    def test_linearlab_filename_exists(self):
        assert os.path.isfile(colorbar._linearlab_filename())


class TestRange(object):

    @pytest.mark.parametrize('cbrange, expected',
                             [([1.5, 1.75], []),
                              ([1.5, 4], [2, 3]),
                              ([1.5, 8], [2, 3, 6]),
                              ([1.5, 18], [2, 3, 6, 10]),
                              ([1.5, 45], [2, 3, 6, 10, 20, 30]),
                              ([2, 8], [2, 3, 6]),
                              ([3, 25], [3, 6, 10, 20])
                              ])
    def test_log_cbticks(self, cbrange, expected):
        assert (colorbar._log_cbticks(cbrange) == expected).all()
        assert (colorbar._set_cbticks(cbrange, {'log_cb': True})[1] == expected).all()

    @pytest.mark.parametrize('value, expected',
                             [(1, '1'),
                              (10, '10'),
                              (100, '100'),
                              (150, '150'),
                              (1000, '1e3'),
                              (1500, '1e3'),
                              (10000, '1e4'),
                              (0.1, '0.1'),
                              (0.11, '0.1')])
    def test_log_tick_format(self, value, expected):
        assert colorbar._log_tick_format(value) == expected

    @pytest.mark.parametrize('d, cbrange',
                             [({'vmin': 0, 'vmax': 1}, None),
                              ({'vmax': 1}, [0, 2]),
                              ({'vmin': 0}, [0.1, 1]),
                              ({}, [0, 1])
                              ])
    def test_set_vmin_vmax(self, d, cbrange):
        expected = {'vmin': 0, 'vmax': 1}
        assert colorbar._set_vmin_vmax(d, cbrange) == expected

    @pytest.mark.parametrize('image, sigma, expected',
                             [(np.ma.array([1, 3, 7, 13, 29, 35], mask=[0, 1, 1, 0, 0, 0]), 1,
                               [13, 29])])
    def test_cbrange_sigma_clip(self, image, sigma, expected):
        assert colorbar._cbrange_sigma_clip(image, sigma=sigma) == expected
