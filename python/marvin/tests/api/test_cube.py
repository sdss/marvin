# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-12 23:40:36
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-19 12:44:46

from __future__ import print_function, division, absolute_import
from marvin.tests.api import MarvinAPITester
import unittest


class TestCubeView(MarvinAPITester):

    def test_get_cube_success(self):
        url = self.get_url('CubeView:index')
        self._load_page('get', url, params={'release': 'MPL-5'})
        data = 'this is a cube'
        self._assert_success(data)


class TestGetCube(TestCubeView):

    def test_get_plateifu_no_release(self):
        errmsg = 'Missing data for required field.'
        url = self.get_url('getCube')
        self._route_no_valid_params(url, 'release', errmsg=errmsg)

    def test_post_plateifu_no_release(self):
        errmsg = 'Missing data for required field.'
        url = self.get_url('getCube')
        self._route_no_valid_params(url, 'release', 'post', errmsg=errmsg)

    def test_post_plateifu_bad_name(self):
        errmsg = 'String does not match expected pattern.'
        url = self.get_url('getCube').format(name='badname')
        self._route_no_valid_params(url, 'name', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_plateifu_short_name(self):
        errmsg = 'Shorter than minimum length 4.'
        url = self.get_url('getCube').format(name='84')
        self._route_no_valid_params(url, 'name', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def _plateifu_success(self, reqtype):
        url = self.get_url('getCube').format(name=self.plateifu)
        data = {'plateifu': self.plateifu, 'mangaid': self.mangaid, 'ra': self.ra, 'dec': self.dec,
                'redshift': self.redshift}
        self._load_page(reqtype, url, params={'release': 'MPL-5'})
        self._assert_success(data)

    def test_get_plateifu_success(self):
        self._plateifu_success('get')

    def test_post_plateifu_success(self):
        self._plateifu_success('post')


if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
