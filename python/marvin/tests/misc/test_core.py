#!/usr/bin/env python
# encoding: utf-8
#
# test_core.py
#
# Created by José Sánchez-Gallego on 1 Dec 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import marvin
import marvin.tests
import marvin.tools.cube


class TestCore(marvin.tests.MarvinTest):
    """A series of tests for MarvinToolsClass."""

    @classmethod
    def setUpClass(cls):
        super(TestCore, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self._reset_the_config()
        self.set_sasurl('local')
        self._update_release('MPL-5')

    def tearDown(self):
        pass

    def test_custom_drpall(self):
        """Tests that drpall is reset when we instantiate a Cube with custom release."""

        self.assertIn('drpall-v2_0_1.fits', marvin.config.drpall)

        cube = marvin.tools.cube.Cube(plateifu=self.plateifu, release='MPL-4')

        self.assertEqual(cube._release, 'MPL-4')
        self.assertEqual(cube._drpver, 'v1_5_1')
        self.assertTrue(os.path.exists(cube._drpall))
        self.assertIn('drpall-v1_5_1.fits', cube._drpall)
        self.assertIn('drpall-v2_0_1.fits', marvin.config.drpall)
