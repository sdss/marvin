# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-07 13:48:11
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-05-18 13:28:30

from __future__ import print_function, division, absolute_import
from marvin.tests.web.conftest import Page
from marvin import config
from marvin.web import create_app
from marvin.api.base import arg_validate as av
import pytest
import os

releases = ['MPL-5']


@pytest.fixture(scope='session', params=releases)
def release(request):
    return request.param


@pytest.fixture(scope='session')
def drpver(release):
    drpver, dapver = config.lookUpVersions(release)
    return drpver


@pytest.fixture(scope='session')
def dapver(release):
    drpver, dapver = config.lookUpVersions(release)
    return dapver


@pytest.fixture(scope='session')
def mode():
    return config.mode


@pytest.fixture(scope='session')
def app():
    app = create_app(debug=True, local=True, use_profiler=False)
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False
    print('init api api')
    return app


@pytest.fixture(scope='session')
def init_api(urlmap):
    #set_sasurl('local')
    config.urlmap = urlmap
    #av.urlmap = urlmap
    config.forceDbOn()


class ApiPage(Page):

    def __init__(self, client, blue, endpoint):
        super(ApiPage, self).__init__(client, blue, endpoint)
        self.api_base_success = dict(status=1, error=None, traceback=None)

    def assert_success(self, expdata, keys=None):
        self.assert200(message='response status should be 200 for ok and not {0}'.format(self.response.status_code))
        assert self.json['status'] == 1
        self.assert_dict_contains_subset(self.api_base_success, self.json)
        if isinstance(expdata, str):
            assert expdata in self.json['data']
        elif isinstance(expdata, dict):
            if keys:
                assert set(expdata.keys()) == set(self.json['data'].keys())
            else:
                assert expdata.items() <= self.json['data'].items()
        elif isinstance(expdata, list):
            assert expdata == self.json['data'], 'two lists should be the same'

    def route_no_valid_params(self, url, noparam, reqtype='get', params=None, errmsg=None):
        self.load_page(reqtype, url, params=params)
        self.assert422(message='response status should be 422 for invalid params')
        assert 'validation_errors' in self.json.keys()
        noparam = [noparam] if not isinstance(noparam, list) else noparam
        invalid = {p: [errmsg] for p in noparam}
        self.assert_dict_contains_subset(invalid, self.json['validation_errors'])

