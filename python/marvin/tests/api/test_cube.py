# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-12 23:40:36
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-13 00:50:13

from __future__ import print_function, division, absolute_import
from marvin.tests.api import MarvinAPITester
import unittest


class TestCubeView(MarvinAPITester):

    def test_get_cube_success(self):
        url = self.get_url('CubeView:index')
        self._load_page('get', url, params={'release': 'MPL-5'})
        data = 'this is a cube'
        self._assert_success(data)

    def test_get_plateifu_no_release(self):
        url = self.get_url('getCube')
        self._route_no_valid_params(url, 'release')

    def test_post_plateifu_no_release(self):
        url = self.get_url('getCube')
        self._route_no_valid_params(url, 'release', 'post')

    def test_post_plateifu_no_name(self):
        url = self.get_url('getCube')
        self._route_no_valid_params(url, 'name', 'post', params={'release': 'MPL-5'})


if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
