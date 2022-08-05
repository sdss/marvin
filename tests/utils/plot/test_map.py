#!/usr/bin/env python
# encoding: utf-8
#
# test_maps.py
#
# @Author: Brett Andrews <andrews>
# @Date:   2017-05-01 09:07:00

# @Last modified by:   andrews
# @Last modified time: 2018-08-07 16:08:71

import matplotlib
import numpy as np
import pytest

import marvin.utils.plot.map as mapplot
from marvin import config
from tests import marvin_test_if
from marvin.utils.datamodel.dap import datamodel

matplotlib_2 = pytest.mark.skipif(int(matplotlib.__version__.split('.')[0]) <= 1,
                                  reason='matplotlib-2.0 or higher required')

value = np.array([[1, -2, 3],
                  [-4, 5, 6],
                  [7, 8, 9]])

ivar = np.array([[4, 10, 1],
                 [0.01, 0.04, 0.0001],
                 [100000, 0, 1]])

ivar_simple = np.array([[0, 0, 0],
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

nocov_simple = np.array([[1, 1, 1],
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

neg_val = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]])

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
    params = datamodel[config.lookUpVersions()[1]].get_plot_params(request.param)
    return params['bitmasks']


class TestMasks(object):

    @pytest.mark.parametrize('ivar, expected',
                             [(ivar_simple, nocov_simple)])
    def test_mask_nocov_ivar(self, ivar, expected):
        actual = mapplot._mask_nocov(mask=None, dapmap=None, ivar=ivar)
        assert np.all(actual == expected)

    @marvin_test_if(mark='skip', maps_release_only=dict(release='DR17'))
    def test_mask_nocov_dapmap(self, maps_release_only):
        ha = maps_release_only['emline_gvel_ha_6564']
        actual = mapplot._mask_nocov(mask=ha.mask, dapmap=ha, ivar=ha.ivar)
        expected = (ha.mask & 2**0 > 0)
        assert np.all(actual == expected)

    @pytest.mark.parametrize('value, ivar, snr_min, expected',
                             [(value, ivar, None, snr_min_none),
                              (value, ivar, 0, snr_min_0),
                              (value, ivar, 1, snr_min_1),
                              (value, ivar, 3, snr_min_3),
                              (value, None, None, np.zeros(value.shape, dtype=bool)),
                              (value, None, 1, np.zeros(value.shape, dtype=bool))])
    def test_mask_low_snr(self, value, ivar, snr_min, expected):
        actual = mapplot.mask_low_snr(value, ivar, snr_min)
        assert np.all(actual == expected)

    @pytest.mark.parametrize('value, expected', [(value, neg_val)])
    def test_mask_neg_values(self, value, expected):
        actual = mapplot.mask_neg_values(value)
        assert np.all(actual == expected)

    @pytest.mark.parametrize('use_masks, mask, expected',
                             [(False, mask_daplike, []),
                              (False, None, []),
                              (True, None, []),
                              (['DONOTUSE'], None, []),
                              (True, mask_daplike, []),
                              (['DONOTUSE'], mask_daplike, ['DONOTUSE'])
                              ])
    def test_format_use_masks_mpl4(self, use_masks, mask, expected, set_release):

        # if config.release != 'MPL-4':
        #     pytest.skip('Only include MPL-4.')

        for prop in ['stellar_vel', 'stellar_sigma', 'emline_gflux', 'specindex']:
            params = datamodel[config.lookUpVersions()[1]].get_plot_params(prop)
            actual = mapplot._format_use_masks(use_masks, mask, dapmap=None,
                                               default_masks=params['bitmasks'])
            assert actual == expected

    @pytest.mark.parametrize('use_masks, mask, expected',
                             [(False, mask_daplike, []),
                              (False, None, []),
                              (True, None, []),
                              (['DONOTUSE'], None, []),
                              (True, mask_daplike, []),
                              (['LOWCOV', 'DONOTUSE'], mask_daplike, ['LOWCOV', 'DONOTUSE'])])
    def test_format_use_masks(self, use_masks, mask, expected, set_release):

        if config.release == 'MPL-4':
            pytest.skip('Skip MPL-4.')

        for prop in ['stellar_vel', 'stellar_sigma', 'emline_gflux', 'specindex']:
            params = datamodel[config.lookUpVersions()[1]].get_plot_params(prop)
            actual = mapplot._format_use_masks(use_masks, mask, dapmap=None,
                                               default_masks=params['bitmasks'])
            assert actual == expected


class TestMapPlot(object):

    @pytest.mark.parametrize('cube_size, sky_coords, expected',
                             [([36, 36], True, np.array([-9, 9, -9, 9])),
                              ([35, 35], True, np.array([-8.75, 8.75, -8.75, 8.75])),
                              ([36, 36], False, np.array([0, 35, 0, 35]))])
    def test_set_extent(self, cube_size, sky_coords, expected):
        extent = mapplot._set_extent(cube_size, sky_coords)
        assert np.all(extent == expected)

    @matplotlib_2
    def test_ax_facecolor(self):
        fig, ax = mapplot._ax_setup(sky_coords=True, fig=None, ax=None, facecolor='#A8A8A8')
        assert ax.get_facecolor() == (0.6588235294117647, 0.6588235294117647, 0.6588235294117647,
                                      1.0)

    @pytest.mark.parametrize('sky_coords, expected',
                             [(True, 'arcsec'),
                              (False, 'spaxel')])
    def test_ax_labels(self, sky_coords, expected):
        fig, ax = mapplot._ax_setup(sky_coords, fig=None, ax=None, facecolor='#A8A8A8')
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
    def test_plot_dapmap(self, maps_release_only, property_name, channel):
        map_ = maps_release_only.getMap(property_name, channel=channel)
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

    def test_plot_matplotlib_rc_restore(self, maps_release_only):
        rcparams = matplotlib.rc_params()
        map_ = maps_release_only.getMap('emline_gflux', channel='ha_6564')
        fig, ax = mapplot.plot(dapmap=map_)
        assert rcparams == matplotlib.rc_params()

    def test_plot_matplotlib_style_sheet(self, maps_release_only):
        map_ = maps_release_only.getMap('emline_gflux', channel='ha_6564')
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

    def test_return_cb(self, maps_release_only):
        map_ = maps_release_only.getMap('emline_gflux', channel='ha_6564')
        fig, ax, cb = mapplot.plot(dapmap=map_, return_cb=True)
        assert isinstance(cb, matplotlib.colorbar.Colorbar)

    def test_return_cbrange(self, maps_release_only):
        map_ = maps_release_only.getMap('emline_gflux', channel='ha_6564')
        cbrange = mapplot.plot(dapmap=map_, return_cbrange=True)
        assert isinstance(cbrange, list)

    def test_symmetric_log_colorbar_error(self, maps_release_only):
        map_ = maps_release_only.getMap('emline_gflux', channel='ha_6564')

        with pytest.raises(AssertionError) as ee:
            fig, ax = mapplot.plot(dapmap=map_, symmetric=True, log_cb=True)

        assert ee.value.args[0] == 'Colorbar cannot be both symmetric and logarithmic.'
