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

from matplotlib import pyplot as plt
import numpy as np

from marvin import config
from marvin.tools.maps import Maps


class TestBPT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.filename_8485_1901_mpl5_spx = os.path.join(
            os.getenv('MANGA_SPECTRO_ANALYSIS'), 'v2_0_1', '2.0.2',
            'SPX-GAU-MILESHC', '8485', '1901', 'manga-8485-1901-MAPS-SPX-GAU-MILESHC.fits.gz')

        config.setMPL('MPL-5')
        config.switchSasUrl('local')
        config.use_sentry = False
        config.add_github_message = False

        cls.emission_mechanisms = ['sf', 'comp', 'agn', 'seyfert', 'liner', 'invalid', 'ambiguous']

    def _run_tests_8485_1901(self, maps):

        masks, figure = maps.get_bpt(show_plot=False, return_figure=True, use_oi=True)
        self.assertIsInstance(figure, plt.Figure)

        for em_mech in self.emission_mechanisms:
            self.assertIn(em_mech, masks.keys())

        self.assertEqual(np.sum(masks['sf']['global']), 62)
        self.assertEqual(np.sum(masks['comp']['global']), 28)

        for em_mech in ['agn', 'seyfert', 'liner']:
            self.assertEqual(np.sum(masks[em_mech]['global']), 0)

        self.assertEqual(np.sum(masks['ambiguous']['global']), 4)
        self.assertEqual(np.sum(masks['invalid']['global']), 1085)

    def test_8485_1901_bpt_file(self):

        maps = Maps(filename=self.filename_8485_1901_mpl5_spx)
        self._run_tests_8485_1901(maps)

    def test_8485_1901_bpt_db(self):

        maps = Maps(plateifu='8485-1901')
        self._run_tests_8485_1901(maps)

    def test_8485_1901_bpt_api(self):

        maps = Maps(plateifu='8485-1901', mode='remote')
        self._run_tests_8485_1901(maps)


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
