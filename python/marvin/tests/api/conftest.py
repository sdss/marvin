# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-07 13:48:11
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-05-07 14:45:55

from __future__ import print_function, division, absolute_import
from marvin.tests.web.conftest import Page, set_sasurl
from marvin import config
from marvin.web import create_app
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
def app():
    app = create_app(debug=True, local=True, use_profiler=False)
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False
    return app


class ApiPage(Page):

    def __init__(self, client, blue, endpoint):
        super(ApiPage, self).__init__(client, blue, endpoint)
        self.api_base_success = dict(status=1, error=None, traceback=None)

    def assert_success(self, expdata, keys=None):
        self.assert200(message='response status should be 200 for ok and not {0}'.format(self.response.status_code))
        assert self.json['status'] == 1
        #print('json', self.json)
        #assert self.api_base_success.items() <= self.json.items()
        self.assert_dict_contains_subset(self.api_base_success, self.json)
        if isinstance(expdata, str):
            assert expdata in self.json['data']
        elif isinstance(expdata, dict):
            if keys:
                assert expdata == self.json['data'].keys()
            else:
                assert expdata.items() <= self.json['data'].items()

    def route_no_valid_params(self, url, noparam, reqtype='get', params=None, errmsg=None):
        self.load_page(reqtype, url, params=params)
        self.assert422(message='response status should be 422 for invalid params')
        assert 'validation_errors' in self.json.keys()
        noparam = [noparam] if not isinstance(noparam, list) else noparam
        invalid = {p: [errmsg] for p in noparam}
        #assert invalid.items() <= self.json['validation_errors'].items()
        self.assert_dict_contains_subset(invalid, self.json['validation_errors'])

