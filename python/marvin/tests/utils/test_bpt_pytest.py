#!/usr/bin/env python
# encoding: utf-8
#
# test_bpt_pytest.py
#
# Created by José Sánchez-Gallego on 10 Feb 2017.
# Converted to pytest by Brett Andrews on 15 Jun 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import warnings

from matplotlib import pyplot as plt
import numpy as np

from marvin.tools.maps import Maps
from marvin.tests import UseReleases, UseBintypes


@UseBintypes('SPX')
@UseReleases('MPL-5')
class TestBPT(object):

    emission_mechanisms = ['sf', 'comp', 'agn', 'seyfert', 'liner', 'invalid', 'ambiguous']

#     def setUp(self):
#         self._reset_the_config()
#         self._update_release('MPL-5')
#         self.set_sasurl('local')
#         self.filename_8485_1901_mpl5_spx = os.path.join(
#             self.mangaanalysis, self.drpver, self.dapver,
#             'SPX-GAU-MILESHC', str(self.plate), self.ifu, self.mapsname)

    def _run_tests_8485_1901(self, maps):

        masks, figure = maps.get_bpt(show_plot=False, return_figure=True, use_oi=True)
        assert isinstance(figure, plt.Figure)

        for em_mech in self.emission_mechanisms:
            assert em_mech in masks.keys()

        assert np.sum(masks['sf']['global']) == 62
        assert np.sum(masks['comp']['global']) == 1

        for em_mech in ['agn', 'seyfert', 'liner']:
            assert np.sum(masks[em_mech]['global']) == 0

        assert np.sum(masks['ambiguous']['global']) == 8
        assert np.sum(masks['invalid']['global']) == 1085

        assert np.sum(masks['sf']['sii']) == 176

    def test_8485_1901_bpt_file(self, galaxy):

        maps = Maps(filename=galaxy.mapspath)
        self._run_tests_8485_1901(maps)

    def test_8485_1901_bpt_db(self, galaxy):

        maps = Maps(plateifu=galaxy.plateifu)
        self._run_tests_8485_1901(maps)

    def test_8485_1901_bpt_api(self, galaxy):

        maps = Maps(plateifu=galaxy.plateifu, mode='remote')
        self._run_tests_8485_1901(maps)

    def test_8485_1901_bpt_no_oi(self, galaxy):

        maps = Maps(plateifu=galaxy.plateifu)
        masks, figure = maps.get_bpt(show_plot=False, return_figure=True, use_oi=False)
        assert isinstance(figure, plt.Figure)

        for em_mech in self.emission_mechanisms:
            assert em_mech in masks.keys()

        assert 'oi' not in masks['sf'].keys()

        assert np.sum(masks['sf']['global']) == 149
        assert np.sum(masks['sf']['sii']) == 176

    def test_8485_1901_bpt_no_figure(self, galaxy):

        maps = Maps(plateifu=galaxy.plateifu)
        bpt_return = maps.get_bpt(show_plot=False, return_figure=False, use_oi=False)

        assert isinstance(bpt_return, dict)

    def test_8485_1901_bpt_snr_min(self, galaxy):

        maps = Maps(plateifu=galaxy.plateifu)
        masks = maps.get_bpt(snr_min=5, return_figure=False, show_plot=False)

        for em_mech in self.emission_mechanisms:
            assert em_mech in masks.keys()

        assert np.sum(masks['sf']['global']) == 28
        assert np.sum(masks['sf']['sii']) == 112

    def test_8485_1901_bpt_snr_deprecated(self, galaxy):

        maps = Maps(plateifu=galaxy.plateifu)

        with warnings.catch_warnings(record=True) as warning_list:
            masks = maps.get_bpt(snr=5, return_figure=False, show_plot=False)

        assert len(warning_list) == 1
        assert str(warning_list[0].message) == \
                         'snr is deprecated. Use snr_min instead. ' \
                         'snr will be removed in a future version of marvin'

        for em_mech in self.emission_mechanisms:
            assert em_mech in masks.keys()

        assert np.sum(masks['sf']['global']) == 28
        assert np.sum(masks['sf']['sii']) == 112
