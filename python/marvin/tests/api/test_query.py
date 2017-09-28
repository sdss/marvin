# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-07 16:40:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-28 17:17:58

from __future__ import print_function, division, absolute_import
from marvin.tests.api.conftest import ApiPage
from marvin.tools.query import Query
from marvin.utils.datamodel.query.base import bestparams
import pytest
import yaml
import os

# @pytest.fixture()
# def page(client, request, init_api):
#     blue, endpoint = request.param
#     page = ApiPage(client, 'api', endpoint)
#     yield page

query_data = yaml.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/query_test_data.dat'))))


@pytest.fixture()
def params(release):
    return {'release': release}


@pytest.fixture()
def data(release):
    return query_data[release]


def get_query_params(release, paramdisplay):
    q = Query(mode='local', release=release)
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
    def test_query_success(self, page, params, reqtype, searchfilter, data):
        params.update({'searchfilter': searchfilter})
        expdata = data['queries'][searchfilter]['top5']
        page.load_page(reqtype, page.url, params=params)
        page.assert_success(expdata, issubset=True)

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
    def test_getparams_success(self, release, page, params, reqtype, paramdisplay):
        params.update({'paramdisplay': paramdisplay})
        expdata = get_query_params(release, paramdisplay)
        page.load_page(reqtype, page.url, params=params)
        page.assert_success(expdata, keys=True)

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('name, missing, errmsg', [(None, 'release', 'Missing data for required field.'),
                                                       (None, 'paramdisplay', 'Missing data for required field.')],
                             ids=['norelease', 'noparamdisplay'])
    def test_plateifu_failure(self, page, reqtype, params, name, missing, errmsg):
        page.route_no_valid_params(page.url, missing, reqtype=reqtype, errmsg=errmsg)


