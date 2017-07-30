# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-19 16:08:47
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-07-30 19:12:14

from __future__ import print_function, division, absolute_import
from marvin.tests.api.conftest import ApiPage
import pytest


@pytest.mark.parametrize('page', [('api', 'PlateView:index')], ids=['plate'], indirect=True)
class TestPlateView(object):

    def test_get_map_success(self, page, params):
        page.load_page('get', page.url, params=params)
        data = 'this is a plate'
        page.assert_success(data)


@pytest.mark.parametrize('page', [('api', 'getPlate')], ids=['getPlate'], indirect=True)
class TestGetPlate(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    def test_plate_success(self, galaxy, page, params, reqtype):
        params.update({'plateid': galaxy.plate})
        data = {'plateid': str(galaxy.plate)}
        page.load_page(reqtype, page.url.format(**params), params=params)
        page.assert_success(data)
        assert data['plateid'] == page.json['data']['plateid']

    @pytest.mark.parametrize('plateid, missing, errmsg',
                             [(None, 'release', 'Missing data for required field.'),
                              ('5000', 'plateid', 'Plateid must be > 6500'),
                              ('84', 'plateid', ['Length must be between 4 and 5.', 'Plateid must be > 6500'])],
                             ids=['norelease', 'badplate', 'shortplate'])
    def test_plate_failures(self, galaxy, page, params, plateid, missing, errmsg):
        params.update({'plateid': plateid})
        if plateid is None:
            page.route_no_valid_params(page.url.format(plateid=galaxy.plate), missing, reqtype='post', errmsg=errmsg)
        else:
            url = page.url.format(**params)
            page.route_no_valid_params(url, missing, reqtype='post', params=params, errmsg=errmsg)


@pytest.mark.parametrize('page', [('api', 'getPlateCubes')], ids=['getPlateCubes'], indirect=True)
class TestGetPlateCubes(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    def test_plate_success(self, galaxy, page, params, reqtype):
        params.update({'plateid': galaxy.plate})
        #data = {"plateifus": ["8485-1902", "8485-12702", "8485-12701", "8485-1901"]}
        data = {'plateifus': [galaxy.plateifu]}
        page.load_page(reqtype, page.url.format(**params), params=params)
        page.assert_success(data)

    @pytest.mark.parametrize('plateid, missing, errmsg',
                             [(None, 'release', 'Missing data for required field.'),
                              ('5000', 'plateid', 'Plateid must be > 6500'),
                              ('84', 'plateid', ['Length must be between 4 and 5.', 'Plateid must be > 6500'])],
                             ids=['norelease', 'badplate', 'shortplate'])
    def test_plate_failures(self, galaxy, page, params, plateid, missing, errmsg):
        params.update({'plateid': plateid})
        if plateid is None:
            page.route_no_valid_params(page.url.format(plateid=galaxy.plate), missing, reqtype='post', errmsg=errmsg)
        else:
            url = page.url.format(**params)
            page.route_no_valid_params(url, missing, reqtype='post', params=params, errmsg=errmsg)

