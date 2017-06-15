#!/usr/bin/env python
# encoding: utf-8
#
# test_bpt.py
#
# Created by José Sánchez-Gallego on 10 Feb 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import os
import unittest
import warnings

from matplotlib import pyplot as plt
import numpy as np

from marvin.tests import MarvinTest
from marvin.tools.maps import Maps
from marvin.core.exceptions import MarvinDeprecationWarning


class TestBPT(MarvinTest):

    @classmethod
    def setUpClass(cls):
        super(TestBPT, cls).setUpClass()

        cls.emission_mechanisms = ['sf', 'comp', 'agn', 'seyfert', 'liner', 'invalid', 'ambiguous']

    def setUp(self):
        self._reset_the_config()
        self._update_release('MPL-5')
        self.set_sasurl('local')
        self.filename_8485_1901_mpl5_spx = os.path.join(
            self.mangaanalysis, self.drpver, self.dapver,
            'SPX-GAU-MILESHC', str(self.plate), self.ifu, self.mapsname)

    def _run_tests_8485_1901(self, maps):

        masks, figure = maps.get_bpt(show_plot=False, return_figure=True, use_oi=True)
        self.assertIsInstance(figure, plt.Figure)

        for em_mech in self.emission_mechanisms:
            self.assertIn(em_mech, masks.keys())

        self.assertEqual(np.sum(masks['sf']['global']), 62)
        self.assertEqual(np.sum(masks['comp']['global']), 1)

        for em_mech in ['agn', 'seyfert', 'liner']:
            self.assertEqual(np.sum(masks[em_mech]['global']), 0)

        self.assertEqual(np.sum(masks['ambiguous']['global']), 8)
        self.assertEqual(np.sum(masks['invalid']['global']), 1085)

        self.assertEqual(np.sum(masks['sf']['sii']), 176)

    def test_8485_1901_bpt_file(self):

        maps = Maps(filename=self.filename_8485_1901_mpl5_spx)
        self._run_tests_8485_1901(maps)

    def test_8485_1901_bpt_db(self):

        maps = Maps(plateifu=self.plateifu)
        self._run_tests_8485_1901(maps)

    def test_8485_1901_bpt_api(self):

        maps = Maps(plateifu=self.plateifu, mode='remote')
        self._run_tests_8485_1901(maps)

    def test_8485_1901_bpt_no_oi(self):

        maps = Maps(plateifu=self.plateifu)
        masks, figure = maps.get_bpt(show_plot=False, return_figure=True, use_oi=False)
        self.assertIsInstance(figure, plt.Figure)

        for em_mech in self.emission_mechanisms:
            self.assertIn(em_mech, masks.keys())

        self.assertNotIn('oi', masks['sf'].keys())

        self.assertEqual(np.sum(masks['sf']['global']), 149)
        self.assertEqual(np.sum(masks['sf']['sii']), 176)

    def test_8485_1901_bpt_no_figure(self):

        maps = Maps(plateifu=self.plateifu)
        bpt_return = maps.get_bpt(show_plot=False, return_figure=False, use_oi=False)

        self.assertIsInstance(bpt_return, dict)

    def test_8485_1901_bpt_snr_min(self):

        maps = Maps(plateifu=self.plateifu)
        masks = maps.get_bpt(snr_min=5, return_figure=False, show_plot=False)

        for em_mech in self.emission_mechanisms:
            self.assertIn(em_mech, masks.keys())

        self.assertEqual(np.sum(masks['sf']['global']), 28)
        self.assertEqual(np.sum(masks['sf']['sii']), 112)

    def test_8485_1901_bpt_snr_deprecated(self):

        maps = Maps(plateifu=self.plateifu)

        with warnings.catch_warnings(record=True) as warning_list:
            masks = maps.get_bpt(snr=5, return_figure=False, show_plot=False)

        self.assertTrue(len(warning_list) == 1)
        self.assertEqual(str(warning_list[0].message),
                         'snr is deprecated. Use snr_min instead. '
                         'snr will be removed in a future version of marvin')

        for em_mech in self.emission_mechanisms:
            self.assertIn(em_mech, masks.keys())

        self.assertEqual(np.sum(masks['sf']['global']), 28)
        self.assertEqual(np.sum(masks['sf']['sii']), 112)


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
