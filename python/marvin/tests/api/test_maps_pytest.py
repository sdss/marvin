# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-07 14:58:52
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-05-07 15:40:25

from __future__ import print_function, division, absolute_import
from marvin.tests.api.conftest import ApiPage
import pytest


@pytest.fixture()
def page(client, request):
    blue, endpoint = request.param
    page = ApiPage(client, 'api', endpoint)
    yield page


@pytest.fixture()
def params(release):
    return {'release': release}


@pytest.mark.parametrize('page', [('api', 'MapsView:index')], ids=['maps'], indirect=True)
class TestMapsView(object):

    def test_get_map_success(self, page, params):
        page.load_page('get', page.url, params=params)
        data = 'this is a maps'
        page.assert_success(data)


@pytest.mark.parametrize('page', [('api', 'getmap')], ids=['getmap'], indirect=True)
class TestGetSingleMap(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('prop, channel', [('emline_sew', 'ha_6564'), ('stellar_vel', None)])
    def test_map_success(self, galaxy, page, params, reqtype, prop, channel):
        params.update({'name': galaxy.plateifu, 'bintype': galaxy.bintype,
                       'template_kin': galaxy.template, 'property_name': prop, 'channel': channel})
        data = ['header', 'unit', 'value', 'mask', 'ivar']
        page.load_page(reqtype, page.url.format(**params), params=params)
        page.assert_success(data, keys=True)

    @pytest.mark.parametrize('name, missing, errmsg, bintype, template, propname, channel',
                             [(None, 'release', 'Missing data for required field.', None, None, None, None),
                              ('badname', 'name', 'String does not match expected pattern.', None, None, None, None),
                              ('84', 'name', 'Shorter than minimum length 4.', None, None, None, None),
                              ('8485-1901', 'bintype', 'Not a valid choice.', 'SPVOR', 'GAU-MILESHC', None, None),
                              ('8485-1901', 'template_kin', 'Not a valid choice.', 'SPX', 'MILESHC', None, None),
                              ('8485-1901', 'bintype', 'Not a valid choice.', 'STON', 'GAU-MILESHC', None, None),
                              ('8485-1901', 'template_kin', 'Not a valid choice.', 'SPX', 'MILES-THIN', None, None),
                              ('8485-1901', 'property_name', 'Not a valid choice.', 'SPX', 'GAU-MILESHC', 'emline', None),
                              ('8485-1901', 'channel', 'Field may not be null.', 'SPX', 'GAU-MILESHC', 'emline_gflux', None)],
                             ids=['norelease', 'badname', 'shortname', 'badbintype', 'badtemplate',
                                  'wrongmplbintype', 'wrongmpltemplate', 'invalidproperty', 'nochannel'])
    def test_map_failures(self, galaxy, page, params, name, missing, errmsg,
                          bintype, template, propname, channel):
        params.update({'name': name, 'bintype': bintype, 'template_kin': template,
                       'property_name': propname, 'channel': channel})
        if name is None:
            page.route_no_valid_params(page.url, missing, reqtype='post', errmsg=errmsg)
        else:
            url = page.url.format(**params)
            url = url.replace('None/', '') if missing == 'channel' else url
            page.route_no_valid_params(url, missing, reqtype='post', params=params, errmsg=errmsg)


