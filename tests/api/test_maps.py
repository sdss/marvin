# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-07 14:58:52
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-07-06 11:35:21

from __future__ import print_function, division, absolute_import
from tests.api.conftest import ApiPage
import pytest

pytestmark = pytest.mark.uses_web


@pytest.mark.parametrize('page', [('api', 'MapsView:index')], ids=['maps'], indirect=True)
class TestMapsView(object):

    def test_get_map_success(self, page, params):
        page.load_page('get', page.url, params=params)
        data = 'this is a maps'
        page.assert_success(data)


@pytest.mark.parametrize('page', [('api', 'getMaps')], ids=['getMaps'], indirect=True)
class TestGetMaps(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    def test_maps_success(self, galaxy, page, params, reqtype):
        params.update({'name': galaxy.plateifu, 'bintype': galaxy.bintype.name,
                       'template': galaxy.template.name})
        data = {'plateifu': galaxy.plateifu, 'mangaid': galaxy.mangaid,
                'bintype': galaxy.bintype.name,
                'template': galaxy.template.name, 'shape': galaxy.shape}
        page.load_page(reqtype, page.url.format(**params), params=params)
        page.assert_success(data)

    @pytest.mark.parametrize('name, missing, errmsg, bintype, template',
                             [(None, 'release', 'Missing data for required field.', None, None),
                              ('badname', 'name', 'String does not match expected pattern.', None, None),
                              ('84', 'name', 'Shorter than minimum length 4.', None, None),
                              ('8485-1901', 'bintype', 'Must be one of: HYB10, SPX, VOR10.', 'SPVOR', 'MILESHC-MASTARSSP'),
                              ('8485-1901', 'template', 'Must be one of: MILESHC-MASTARHC2, MILESHC-MASTARSSP.', 'SPX', 'MILESHC'),
                              ('8485-1901', 'bintype', 'Must be one of: HYB10, SPX, VOR10.', 'STONY', 'MILESHC-MASTARSSP'),
                              ('8485-1901', 'template', 'Must be one of: MILESHC-MASTARHC2, MILESHC-MASTARSSP.', 'SPX', 'MILES'),
                              ('8485-1901', 'bintype', 'Field may not be null.', None, None),
                              ('8485-1901', 'template', 'Field may not be null.', 'HYB10', None)],
                             ids=['norelease', 'badname', 'shortname', 'badbintype', 'badtemplate',
                                  'wrongmplbintype', 'wrongmpltemplate', 'nobintype', 'notemplate'])
    def test_maps_failures(self, galaxy, page, params, name, missing, errmsg, bintype, template):
        params.update({'name': name, 'bintype': bintype, 'template': template})
        if name is None:
            page.route_no_valid_params(page.url, missing, reqtype='post', errmsg=errmsg)
        else:
            url = page.url.format(**params)
            url = url.replace('None/', '') if missing in ['bintype', 'template'] else url
            page.route_no_valid_params(url, missing, reqtype='post', params=params, errmsg=errmsg)


propmsg = ('Must be one of: spx_skycoo, spx_ellcoo, spx_mflux, spx_snr, binid, bin_lwskycoo, bin_lwellcoo, '
            'bin_area, bin_farea, bin_mflux, bin_snr, stellar_vel, stellar_sigma, specindex, specindex_corr, '
            'emline_sflux, emline_sew, emline_gflux, emline_gvel, emline_gew, emline_gsigma, emline_instsigma, '
            'emline_ga, emline_ganr, emline_lfom, emline_fom, stellar_fom, stellar_sigmacorr, emline_sew_cnt, '
            'emline_gew_cnt, specindex_model, specindex_bf, specindex_bf_corr, specindex_bf_model, specindex_wgt, '
            'specindex_wgt_corr, specindex_wgt_model.')


@pytest.mark.parametrize('page', [('api', 'getmap')], ids=['getmap'], indirect=True)
class TestGetSingleMap(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('prop, channel', [('emline_gflux', 'ha_6564'), ('stellar_vel', None)])
    def test_map_success(self, galaxy, page, params, reqtype, prop, channel):
        params.update({'name': galaxy.plateifu, 'bintype': galaxy.bintype.name,
                       'template': galaxy.template.name, 'property_name': prop, 'channel': channel})
        data = {'unit': '', 'value': '', 'mask': '', 'ivar': ''}
        page.load_page(reqtype, page.url.format(**params), params=params)
        page.assert_success(data, keys=True)

    @pytest.mark.parametrize('name, missing, errmsg, bintype, template, propname, channel',
                             [(None, 'release', 'Missing data for required field.', None, None, None, None),
                              ('badname', 'name', 'String does not match expected pattern.', None, None, None, None),
                              ('84', 'name', 'Shorter than minimum length 4.', None, None, None, None),
                              ('8485-1901', 'bintype', 'Must be one of: HYB10, SPX, VOR10.', 'SPVOR', 'MILESHC-MASTARSSP', None, None),
                              ('8485-1901', 'template', 'Must be one of: MILESHC-MASTARHC2, MILESHC-MASTARSSP.', 'SPX', 'MILESHC', None, None),
                              ('8485-1901', 'bintype', 'Must be one of: HYB10, SPX, VOR10.', 'STONY', 'MILESHC-MASTARSSP', None, None),
                              ('8485-1901', 'template', 'Must be one of: MILESHC-MASTARHC2, MILESHC-MASTARSSP.', 'SPX', 'MILES', None, None),
                              ('8485-1901', 'property_name', propmsg, 'HYB10', 'MILESHC-MASTARSSP', 'emline', None),
                              ('8485-1901', 'channel', 'Field may not be null.', 'HYB10', 'MILESHC-MASTARSSP', 'emline_gflux', None)],
                             ids=['norelease', 'badname', 'shortname', 'badbintype', 'badtemplate',
                                  'wrongmplbintype', 'wrongmpltemplate', 'invalidproperty', 'nochannel'])
    def test_map_failures(self, galaxy, page, params, name, missing, errmsg,
                          bintype, template, propname, channel):
        params.update({'name': name, 'bintype': bintype, 'template': template,
                       'property_name': propname, 'channel': channel})
        if name is None:
            page.route_no_valid_params(page.url, missing, reqtype='post', errmsg=errmsg)
        else:
            url = page.url.format(**params)
            url = url.replace('None/', '') if missing == 'channel' else url
            page.route_no_valid_params(url, missing, reqtype='post', params=params, errmsg=errmsg)
