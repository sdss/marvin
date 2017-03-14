# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-22 10:38:28
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-03-13 14:14:00

from __future__ import print_function, division, absolute_import
from marvin.tests.web import MarvinWebTester
from marvin.web.controllers.galaxy import make_nsa_dict
from marvin.tools.cube import Cube
from marvin import config
import unittest


class TestGalaxyPage(MarvinWebTester):

    render_templates = False

    def setUp(self):
        super(TestGalaxyPage, self).setUp()
        self.blue = 'galaxy_page'
        config.setRelease('MPL-5')
        self.mode = config.mode
        self.release = config.release
        self.params = {'release': self.release}

        # set up cube and expected values
        self.cube = Cube(plateifu=self.plateifu, mode=self.mode)

        # NSA params for 8485-1901
        self.exp_nsa_plotcols = {'elpetro_absmag_i': -19.1125469207764,
                                 'elpetro_mag_g_r': 0.64608402745868077, 'z': 0.0407447,
                                 'elpetro_th50_r': 1.33067, 'elpetro_logmass': 9.565475912843823,
                                 'elpetro_ba': 0.87454, 'elpetro_mag_i_z': 0.2311751372102151,
                                 'elpetro_phi': 154.873, 'elpetro_mtol_i': 1.30610692501068,
                                 'elpetro_th90_r': 3.6882, 'elpetro_mag_u_r': 1.8892372699482216,
                                 'sersic_n': 3.29617}

    def test_assert_galaxy_template_used(self):
        url = self.get_url('Galaxy:index')
        self._load_page('get', url)
        self.assertEqual('', self.data)
        self.assert_template_used('galaxy.html')


class TestNSA(TestGalaxyPage):

    def test_nsadict_correct(self):
        nsa, cols = make_nsa_dict(self.cube.nsa)
        self.assertDictContainsSubset(self.exp_nsa_plotcols, nsa)
        self.assertListIn(self.exp_nsa_plotcols.keys(), cols)

    def test_initnsa_method_not_allowed(self):
        url = self.get_url('initnsaplot')
        self._load_page('get', url, params=self.params)
        self.assert_template_used('errors/method_not_allowed.html')

    def test_initnsa_no_plateifu(self):
        errmsg = 'Field may not be null.'
        url = self.get_url('initnsaplot')
        self._route_no_valid_webparams(url, 'plateifu', reqtype='post', errmsg=errmsg)

if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
