#!/usr/bin/env python
# encoding: utf-8
#
# test_maps.py
#
# @Author: Brett Andrews <andrews>
# @Date:   2017-05-01 09:07:00
# @Last modified by:   andrews
# @Last modified time: 2017-07-19 10:07:33

import numpy as np
import matplotlib
import pytest

from marvin import config
from marvin.tools.maps import Maps
import marvin.utils.plot.map as mapplot
from marvin.utils.general import get_plot_params


matplotlib_2 = pytest.mark.skipif(int(matplotlib.__version__.split('.')[0]) <= 1,
                                  reason='matplotlib-2.0 or higher required')

value = np.array([[1, -2, 3],
                  [-4, 5, 6],
                  [7, 8, 9]])

ivar = np.array([[4, 10, 1],
                 [0.01, 0.04, 0.0001],
                 [100000, 0, 1]])

ivar_mpl4 = np.array([[0, 0, 0],
                      [0, 3, 0],
                      [0, 0, 0.5]])

mask_simple = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=bool)

mask_binary = np.array([[0b000, 0b001, 0b010],
                        [0b011, 0b100, 0b101],
                        [0b110, 0b111, 0b000]])

mask_daplike = np.array([[0b0000000000000000000000000000000,
                          0b0000000000000000000000000000001,
                          0b0000000000000000000000000100000],
                         [0b1000000000000000000000001000001,
                          0b1000000000000000000000010000000,
                          0b1000000000000000000000000000001],
                         [0b0000000000000000000000000000000,
                          0b1000000000000000000000011100001,
                          0b0000000000000000000000000000000]])

nocov = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]], dtype=bool)

nocov_mpl4 = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=bool)

bad_data = np.array([[0, 0, 1],
                     [1, 1, 1],
                     [0, 1, 0]])

bad_data_mpl4 = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=bool)

# SNR = value * np.sqrt(ivar)
# [[  2.00e+00,  -6.32e+00,   3.00e+00],
#  [ -4.00e-01,   1.00e+00,   6.00e-02],
#  [  2.21e+03,   0.00e+00,   9.00e+00]]

snr_min_none = np.array([[0, 0, 0],
                         [0, 0, 0],
                         [0, 1, 0]])

snr_min_0 = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0]])

snr_min_1 = np.array([[0, 0, 0],
                      [1, 0, 1],
                      [0, 1, 0]])

snr_min_3 = np.array([[1, 0, 0],
                      [1, 1, 1],
                      [0, 1, 0]])

log_cb_true = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0]])

log_cb_false = np.zeros(value.shape, dtype=bool)

image_none_true = np.array([[0, 1, 1],
                            [1, 1, 1],
                            [0, 1, 0]])

image_3_true = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [0, 1, 0]])

image_none_false = np.array([[0, 1, 1],
                             [1, 1, 1],
                             [0, 1, 0]])

image_3_false = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [0, 1, 0]])


@pytest.fixture(scope='module', params=['stellar_vel', 'stellar_sigma', 'emline_gflux',
                                        'specindex'])
def bits(request, set_release):
    params = get_plot_params(dapver=config.lookUpVersions()[1], prop=request.param)
    return params['bitmasks']


class TestMasks(object):

    @pytest.mark.parametrize('mask, ivar, expected',
                             [(mask_simple, ivar_mpl4, nocov_mpl4)])
    def test_no_coverage_mask_mpl4(self, mask, ivar, expected):
        if config.release != 'MPL-4':
            pytest.skip('Use generic test for NOCOV bitmask in MPL-5+.')
        nocov = mapplot.no_coverage_mask(mask=mask, bit=None, ivar=ivar)
        assert np.all(nocov == expected)

    @pytest.mark.parametrize('mask, ivar, expected',
                             [(mask_simple, ivar, nocov),
                              (mask_binary, ivar, nocov),
                              (mask_daplike, ivar, nocov)])
    def test_no_coverage_mask(self, mask, bits, ivar, expected):
        if config.release == 'MPL-4':
            pytest.skip('NOCOV bitmask only exists in MPL-5+')
        nocov = mapplot.no_coverage_mask(mask=mask, bit=bits['nocov'], ivar=ivar)
        assert np.all(nocov == expected)

    @pytest.mark.parametrize('mask, expected', [(mask_simple, bad_data_mpl4)])
    def test_bad_data_mask_mpl4(self, mask, expected):
        if config.release != 'MPL-4':
            pytest.skip('Use generic test for bad data bitmasks in MPL-5+.')
        bad_data = mapplot.bad_data_mask(mask=mask, bits={'doNotUse': 0})
        assert np.all(bad_data == expected)

    @pytest.mark.parametrize('mask, expected', [(mask_daplike, bad_data)])
    def test_bad_data_mask(self, mask, bits, expected):
        if config.release == 'MPL-4':
            pytest.skip('UNRELIABLE bitmask only exists in MPL-5+')
        bad_data = mapplot.bad_data_mask(mask=mask, bits=bits['badData'])
        assert np.all(bad_data == expected)

    @pytest.mark.parametrize('value, ivar, snr_min, expected',
                             [(value, ivar, None, snr_min_none),
                              (value, ivar, 0, snr_min_0),
                              (value, ivar, 1, snr_min_1),
                              (value, ivar, 3, snr_min_3),
                              (value, None, None, np.zeros(value.shape, dtype=bool)),
                              (value, None, 1, np.zeros(value.shape, dtype=bool))])
    def test_low_snr_mask(self, value, ivar, snr_min, expected):
        low_snr = mapplot.low_snr_mask(value, ivar, snr_min)
        assert np.all(low_snr == expected)

    @pytest.mark.parametrize('value, log_cb, expected',
                             [(value, True, log_cb_true),
                              (value, False, log_cb_false)])
    def test_log_colorbar_mask(self, value, log_cb, expected):
        log_colorbar = mapplot.log_colorbar_mask(value, log_cb)
        assert np.all(log_colorbar == expected)

    @pytest.mark.parametrize('value, nocov, bad_data, snr_min, log_cb, expected_mask',
                             [(value, nocov, bad_data, snr_min_none, log_cb_true, image_none_true),
                              (value, nocov, bad_data, snr_min_3, log_cb_true, image_3_true),
                              (value, nocov, bad_data, snr_min_none, log_cb_false,
                               image_none_false),
                              (value, nocov, bad_data, snr_min_3, log_cb_false, image_3_false)])
    def test_select_good_spaxels(self, value, nocov, bad_data, snr_min, log_cb, expected_mask):
        image = mapplot.select_good_spaxels(value, nocov, bad_data, snr_min, log_cb)
        expected = np.ma.array(value, mask=expected_mask)
        assert np.all(image.data == expected.data)
        assert np.all(image.mask == expected.mask)


class TestMapPlot(object):

    @pytest.mark.parametrize('cube_size, sky_coords, expected',
                             [([36, 36], True, np.array([-18, 18, -18, 18])),
                              ([35, 35], True, np.array([-17.5, 17.5, -17.5, 17.5])),
                              ([36, 36], False, np.array([0, 35, 0, 35]))])
    def test_set_extent(self, cube_size, sky_coords, expected):
        extent = mapplot.set_extent(cube_size, sky_coords)
        assert np.all(extent == expected)

    @matplotlib_2
    def test_set_hatch_linewidth(self, maps):
        map_ = maps.getMap('emline_gflux', channel='ha_6564')
        fig, ax = mapplot.plot(dapmap=map_)
        assert matplotlib.rcParams['hatch.linewidth'] == 0.5

    @matplotlib_2
    def test_set_hatch_color(self, maps):
        map_ = maps.getMap('emline_gflux', channel='ha_6564')
        fig, ax = mapplot.plot(dapmap=map_)
        assert matplotlib.rcParams['hatch.color'] == 'w'

    @matplotlib_2
    def test_ax_facecolor(self):
        fig, ax = mapplot.ax_setup(sky_coords=True, fig=None, ax=None, facecolor='#A8A8A8')
        assert ax.get_facecolor() == (0.6588235294117647, 0.6588235294117647, 0.6588235294117647,
                                      1.0)

    @pytest.mark.parametrize('sky_coords, expected',
                             [(True, 'arcsec'),
                              (False, 'spaxel')])
    def test_ax_labels(self, sky_coords, expected):
        fig, ax = mapplot.ax_setup(sky_coords, fig=None, ax=None, facecolor='#A8A8A8')
        assert ax.get_xlabel() == expected
        assert ax.get_ylabel() == expected

    @pytest.mark.parametrize('title, property_name, channel, expected',
                             [('test-title', None, None, 'test-title'),
                              ('test-title', 'prop', 'channel', 'test-title'),
                              (None, 'prop', 'channel', 'prop channel'),
                              (None, 'emline_gflux', 'ha_6564', 'emline gflux ha 6564'),
                              (None, 'prop_flux', None, 'prop flux'),
                              (None, None, 'channel', 'channel'),
                              (None, None, None, '')])
    def test_set_title(self, title, property_name, channel, expected):
        title = mapplot.set_title(title, property_name, channel)
        assert title == expected

    @pytest.mark.parametrize('property_name, channel',
                             [('emline_gflux', 'ha_6564'),
                              ('emline_gflux', 'oiii_5008'),
                              ('stellar_vel', None),
                              ('stellar_sigma', None),
                              ('specindex', 'd4000')])
    def test_plot_dapmap(self, maps, property_name, channel):
        map_ = maps.getMap(property_name, channel=channel)
        fig, ax = mapplot.plot(dapmap=map_)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes._axes.Axes)

    @pytest.mark.parametrize('mask', [(None), (mask_simple), (mask_binary), (mask_daplike)])
    @pytest.mark.parametrize('ivar', [(None), (ivar)])
    @pytest.mark.parametrize('value', [(value)])
    def test_plot_user_defined_map(self, value, ivar, mask):
        fig, ax = mapplot.plot(value=value, ivar=ivar, mask=mask)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes._axes.Axes)

    def test_plot_matplotlib_rc_restore(self, maps):
        rcparams = matplotlib.rc_params()
        map_ = maps.getMap('emline_gflux', channel='ha_6564')
        fig, ax = mapplot.plot(dapmap=map_)
        assert rcparams == matplotlib.rc_params()

    def test_plot_matplotlib_style_sheet(self, maps):
        map_ = maps.getMap('emline_gflux', channel='ha_6564')
        fig, ax = mapplot.plot(dapmap=map_, plt_style='ggplot')
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes._axes.Axes)

    @pytest.mark.parametrize('title, expected',
                             [('emline_gflux_ha_6564', 'default'),
                              ('emline_gvel_ha_6564', 'vel'),
                              ('emline_gsigma_hb_4862', 'sigma'),
                              ('stellar_vel', 'vel'),
                              ('stellar_sigma', 'sigma')])
    def test_get_prop(self, title, expected):
        assert mapplot._get_prop(title) == expected
    
    def test_return_cb(self, maps):
        map_ = maps.getMap('emline_gflux', channel='ha_6564')
        fig, ax, cb = mapplot.plot(dapmap=map_, return_cb=True)
        assert isinstance(cb, matplotlib.colorbar.Colorbar)

