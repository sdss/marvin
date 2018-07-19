# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-19 16:34:31
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-10 14:23:32

from __future__ import print_function, division, absolute_import
from marvin.tests.api.conftest import ApiPage
import pytest


@pytest.mark.parametrize('page', [('api', 'mangaid2plateifu')], ids=['mangaid2plateifu'], indirect=True)
class TestGeneralMangaid2Plateifu(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    def test_getplateifu_success(self, galaxy, page, params, reqtype):
        data = galaxy.plateifu
        page.load_page(reqtype, page.url.format(mangaid=galaxy.mangaid), params=params)
        page.assert_success(data)

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('mangaid', [('1209232')], ids=['badid'])
    def test_getplateifu_noresult(self, mangaid, page, params, reqtype):
        data = None
        error = "manga2plateifu failed with error: no plate-ifus found for mangaid={0}".format(mangaid)
        page.load_page(reqtype, page.url.format(mangaid=mangaid), params=params)
        assert page.json['status'] == -1
        assert page.json['error'] == error

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('mangaid, missing, errmsg', [(None, 'release', 'Missing data for required field.'),
                                                          ('12', 'mangaid', 'Length must be between 4 and 20.')],
                             ids=['norelease', 'shortname'])
    def test_getplateifu_failure(self, galaxy, page, reqtype, params, mangaid, missing, errmsg):
        if mangaid is None:
            page.route_no_valid_params(page.url.format(mangaid=galaxy.mangaid), missing, reqtype=reqtype, errmsg=errmsg)
        else:
            page.route_no_valid_params(page.url.format(mangaid=mangaid), missing, reqtype=reqtype, params=params, errmsg=errmsg)


@pytest.mark.parametrize('page', [('api', 'nsa_full')], ids=['nsa_full'], indirect=True)
class TestGeneralNSAFull(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    def test_getnsa_success(self, galaxy, page, params, reqtype):
        page.load_page(reqtype, page.url.format(mangaid=galaxy.mangaid), params=params)
        page.assert_success(galaxy.nsa_data['nsa'])

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('mangaid', [('1209232')], ids=['badid'])
    def test_getnsa_noresult(self, mangaid, page, params, reqtype):
        error = "get_nsa_data failed with error: get_nsa_data: cannot find NSA row for mangaid={0}".format(mangaid)
        page.load_page(reqtype, page.url.format(mangaid=mangaid), params=params)
        assert page.json['data'] is None
        assert page.json['status'] == -1
        assert page.json['error'] == error


@pytest.mark.parametrize('page', [('api', 'nsa_drpall')], ids=['nsa_drpall'], indirect=True)
class TestGeneralNSADrpall(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    def test_getnsa_success(self, galaxy, page, params, reqtype):
        page.load_page(reqtype, page.url.format(mangaid=galaxy.mangaid), params=params)
        page.assert_success(galaxy.nsa_data['drpall'])

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('mangaid', [('1209232')], ids=['badid'])
    def test_getnsa_noresult(self, mangaid, page, params, reqtype):
        error = "get_nsa_data failed with error: no plate-ifus found for mangaid={0}".format(mangaid)
        page.load_page(reqtype, page.url.format(mangaid=mangaid), params=params)
        assert page.json['data'] is None
        assert page.json['status'] == -1
        assert page.json['error'] == error
