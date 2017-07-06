# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-07 16:40:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-07-06 16:01:17

from __future__ import print_function, division, absolute_import
from marvin.tests.api.conftest import ApiPage
from marvin.tools.query import Query
from marvin.tools.query.query_utils import bestparams
import pytest


# @pytest.fixture()
# def page(client, request, init_api):
#     blue, endpoint = request.param
#     page = ApiPage(client, 'api', endpoint)
#     yield page


@pytest.fixture()
def params(release):
    return {'release': release}


def get_query_params(paramdisplay):
    q = Query(mode='local')
    if paramdisplay == 'best':
        qparams = bestparams
    else:
        qparams = q.get_available_params('all')
    return qparams


@pytest.mark.parametrize('page', [('api', 'QueryView:index')], ids=['query'], indirect=True)
class TestQueryView(object):

    def test_get_query_success(self, page, params):
        page.load_page('get', page.url, params=params)
        data = 'this is a query'
        page.assert_success(data)


@pytest.mark.parametrize('page', [('api', 'querycubes')], ids=['querycubes'], indirect=True)
class TestQueryCubes(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('searchfilter', [('nsa.z < 0.1')])
    def test_query_success(self, page, params, reqtype, searchfilter):
        params.update({'searchfilter': searchfilter})
        data = [["1-209232", 8485, "8485-1901", "1901", 0.0407447],
                ["1-209113", 8485, "8485-1902", "1902", 0.0378877],
                ["1-209191", 8485, "8485-12701", "12701", 0.0234253],
                ["1-209151", 8485, "8485-12702", "12702", 0.0185246]]
        page.load_page(reqtype, page.url, params=params)
        page.assert_success(data)

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('name, missing, errmsg', [(None, 'release', 'Missing data for required field.'),
                                                       (None, 'searchfilter', 'Missing data for required field.')],
                             ids=['norelease', 'nosearchfilter'])
    def test_query_failure(self, page, reqtype, params, name, missing, errmsg):
        page.route_no_valid_params(page.url, missing, reqtype=reqtype, errmsg=errmsg)


@pytest.mark.parametrize('page', [('api', 'getparams')], ids=['getparams'], indirect=True)
class TestQueryGetParams(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('paramdisplay', [('all'), ('best')])
    def test_getparams_success(self, page, params, reqtype, paramdisplay):
        params.update({'paramdisplay': paramdisplay})
        data = get_query_params(paramdisplay)
        page.load_page(reqtype, page.url, params=params)
        page.assert_success(data, keys=True)

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('name, missing, errmsg', [(None, 'release', 'Missing data for required field.'),
                                                       (None, 'paramdisplay', 'Missing data for required field.')],
                             ids=['norelease', 'noparamdisplay'])
    def test_plateifu_failure(self, page, reqtype, params, name, missing, errmsg):
        page.route_no_valid_params(page.url, missing, reqtype=reqtype, errmsg=errmsg)


