# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-03-15 10:00:32
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-03-15 16:14:36

from __future__ import print_function, division, absolute_import
from marvin.tests.api import MarvinAPITester
import unittest


class TestMapsView(MarvinAPITester):

    def test_get_map_success(self):
        url = self.get_url('MapsView:index')
        self._load_page('get', url, params={'release': 'MPL-5'})
        data = 'this is a maps'
        self._assert_success(data)


class TestGetMaps(TestMapsView):

    def test_get_map_no_release(self):
        errmsg = 'Missing data for required field.'
        url = self.get_url('getMaps')
        self._route_no_valid_params(url, 'release', errmsg=errmsg)

    def test_post_map_no_release(self):
        errmsg = 'Missing data for required field.'
        url = self.get_url('getMaps')
        self._route_no_valid_params(url, 'release', 'post', errmsg=errmsg)

    def test_post_map_bad_name(self):
        errmsg = 'String does not match expected pattern.'
        url = self.get_url('getMaps').format(name='badname', bintype=None, template_kin=None)
        self._route_no_valid_params(url, 'name', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_short_name(self):
        errmsg = 'Shorter than minimum length 4.'
        url = self.get_url('getMaps').format(name='84', bintype=None, template_kin=None)
        self._route_no_valid_params(url, 'name', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_invalid_bintype(self):
        errmsg = 'Not a valid choice.'
        url = self.get_url('getMaps').format(name=self.plateifu, bintype='SPVOR', template_kin='GAU-MILESHC')
        self._route_no_valid_params(url, 'bintype', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_invalid_template_kin(self):
        errmsg = 'Not a valid choice.'
        url = self.get_url('getMaps').format(name=self.plateifu, bintype='SPX', template_kin='MILESHC')
        self._route_no_valid_params(url, 'template_kin', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_wrong_mpl_bintype(self):
        errmsg = 'Not a valid choice.'
        url = self.get_url('getMaps').format(name=self.plateifu, bintype='STON', template_kin='GAU-MILESHC')
        self._route_no_valid_params(url, 'bintype', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_wrong_mpl_template_kin(self):
        errmsg = 'Not a valid choice.'
        url = self.get_url('getMaps').format(name=self.plateifu, bintype='SPX', template_kin='MILES-THIN')
        self._route_no_valid_params(url, 'template_kin', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_no_bintype(self):
        errmsg = 'Field may not be null.'
        url = self.get_url('getMaps').format(name=self.plateifu, bintype=None, template_kin=None)
        url = url.replace('None/', '')
        self._route_no_valid_params(url, 'bintype', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_no_template_kin(self):
        errmsg = 'Field may not be null.'
        url = self.get_url('getMaps').format(name=self.plateifu, bintype='SPX', template_kin=None)
        url = url.replace('None/', '')
        self._route_no_valid_params(url, 'template_kin', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def _map_success(self, reqtype):
        url = self.get_url('getMaps').format(name=self.plateifu, bintype=self.defaultbin, template_kin=self.defaulttemp)
        data = {'plateifu': self.plateifu, 'mangaid': self.mangaid, 'bintype': self.defaultbin, 'template_kin': self.defaulttemp,
                'shape': [34, 34]}
        print('the url', url)
        self._load_page(reqtype, url, params={'release': 'MPL-5'})
        self._assert_success(data)

    def test_get_map_success(self):
        self._map_success('get')

    def test_post_map_success(self):
        self._map_success('post')


class TestGetSingleMap(TestMapsView):

    def test_post_map_no_release(self):
        errmsg = 'Missing data for required field.'
        url = self.get_url('getmap')
        self._route_no_valid_params(url, 'release', 'post', errmsg=errmsg)

    def test_post_map_bad_name(self):
        errmsg = 'String does not match expected pattern.'
        params = {'name': 'badname', 'bintype': None, 'template_kin': None, 'property_name': None, 'channel': None}
        url = self.get_url('getmap').format(**params)
        self._route_no_valid_params(url, 'name', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_short_name(self):
        errmsg = 'Shorter than minimum length 4.'
        params = {'name': '84', 'bintype': None, 'template_kin': None, 'property_name': None, 'channel': None}
        url = self.get_url('getmap').format(**params)
        self._route_no_valid_params(url, 'name', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_invalid_bintype(self):
        errmsg = 'Not a valid choice.'
        params = {'name': self.plateifu, 'bintype': 'SPVOR', 'template_kin': 'GAU-MILESHC', 'property_name': None, 'channel': None}
        url = self.get_url('getmap').format(**params)
        self._route_no_valid_params(url, 'bintype', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_invalid_template_kin(self):
        errmsg = 'Not a valid choice.'
        params = {'name': self.plateifu, 'bintype': 'SPX', 'template_kin': 'MILESHC', 'property_name': None, 'channel': None}
        url = self.get_url('getmap').format(**params)
        self._route_no_valid_params(url, 'template_kin', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_wrong_mpl_bintype(self):
        errmsg = 'Not a valid choice.'
        params = {'name': self.plateifu, 'bintype': 'STON', 'template_kin': 'GAU-MILESHC', 'property_name': None, 'channel': None}
        url = self.get_url('getmap').format(**params)
        self._route_no_valid_params(url, 'bintype', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_wrong_mpl_template_kin(self):
        errmsg = 'Not a valid choice.'
        params = {'name': self.plateifu, 'bintype': 'SPX', 'template_kin': 'MILES-THIN', 'property_name': None, 'channel': None}
        url = self.get_url('getmap').format(**params)
        self._route_no_valid_params(url, 'template_kin', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_invalid_property(self):
        errmsg = 'Not a valid choice.'
        params = {'name': self.plateifu, 'bintype': 'SPX', 'template_kin': 'GAU-MILESHC', 'property_name': 'emline', 'channel': None}
        url = self.get_url('getmap').format(**params)
        url = url.replace('None/', '')
        self._route_no_valid_params(url, 'property_name', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def test_post_map_no_channel(self):
        errmsg = 'Field may not be null.'
        params = {'name': self.plateifu, 'bintype': 'SPX', 'template_kin': 'GAU-MILESHC', 'property_name': 'emline_gflux', 'channel': None}
        url = self.get_url('getmap').format(**params)
        url = url.replace('None/', '')
        self._route_no_valid_params(url, 'channel', 'post', params={'release': 'MPL-5'}, errmsg=errmsg)

    def _map_success(self, reqtype, prop, channel):
        params = {'name': self.plateifu, 'bintype': self.defaultbin, 'template_kin': self.defaulttemp, 'property_name': prop, 'channel': channel}
        url = self.get_url('getmap').format(**params)
        data = ['header', 'unit', 'value', 'mask', 'ivar']
        self._load_page(reqtype, url, params={'release': 'MPL-5'})
        self._assert_success(data, keys=True)

    def test_get_map_success(self):
        self._map_success('get', 'emline_sew', 'ha_6564')

    def test_post_map_success_emlinesew(self):
        self._map_success('post', 'emline_sew', 'ha_6564')

    def test_post_map_success_stvel(self):
        self._map_success('post', 'stellar_vel', None)

