# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-07 13:48:11
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-13 18:24:09

from __future__ import print_function, division, absolute_import
from tests.web.conftest import Page
from marvin import config
from marvin.web import create_app
from marvin.web.settings import TestConfig, CustomConfig
from marvin.web.extensions import limiter
from brain.utils.general import build_routemap
import pytest
import six

pytestmark = pytest.mark.uses_web


@pytest.fixture(scope='session')
def app():
    object_config = type('Config', (TestConfig, CustomConfig), dict())
    app = create_app(debug=True, local=True, object_config=object_config)
    limiter.enabled = False
    return app


@pytest.fixture()
def urlmap(app):
    ''' fixture for building a new urlmap without nonsense '''
    urlmap = build_routemap(app)
    config.urlmap = urlmap


@pytest.fixture(scope='function')
def init_api(monkeyauth, set_config, urlmap):
    config.forceDbOn()
    #config.login()


class ApiPage(Page):

    def __init__(self, app, client, blue, endpoint):
        super(ApiPage, self).__init__(app, client, blue, endpoint)
        self.api_base_success = dict(status=1, error=None, traceback=None)

    def assert_success(self, expdata=None, keys=None, issubset=None):
        self.assert200(message='response status should be 200 for ok and not {0}'.format(self.response.status_code))
        assert self.json['status'] == 1
        self.assert_dict_contains_subset(self.api_base_success, self.json)
        if isinstance(expdata, str):
            assert isinstance(self.json['data'], six.string_types), 'response data should be a string'
            assert expdata in self.json['data']
        elif isinstance(expdata, dict):
            assert isinstance(self.json['data'], dict), 'response data should be a dict'
            if keys:
                assert set(expdata.keys()) == set(self.json['data'].keys())
            else:
                #assert expdata.items() <= self.json['data'].items()
                self.assert_dict_contains_subset(expdata, self.json['data'])
        elif isinstance(expdata, list):
            assert isinstance(self.json['data'], list), 'response data should be a list'
            if issubset:
                subset = all(row in self.json['data'] for row in expdata)
                assert subset is True, 'one should be subset of the other'
            else:
                assert expdata == self.json['data'], 'two lists should be the same'

    def route_no_valid_params(self, url, noparam, reqtype='get', params=None, errmsg=None):
        self.load_page(reqtype, url, params=params)
        if noparam == 'release':
            self.assert409(message='Error: data release None no longer supported by the Marvin API.')
        else:
            self.assert422(message='response status should be 422 for invalid params')
            assert 'validation_errors' in self.json.keys()
            noparam = [noparam] if not isinstance(noparam, list) else noparam
            errmsg = [errmsg] if not isinstance(errmsg, list) else errmsg
            invalid = {p: errmsg for p in noparam}
            self.assert_dict_contains_subset(invalid, self.json['validation_errors'])


@pytest.fixture()
def page(app, client, request, init_api):
    blue, endpoint = request.param
    page = ApiPage(app, client, 'api', endpoint)
    yield page
    url = page.get_url('index_page', 'Marvin:clear_session')
    page.load_page('get', url)


@pytest.fixture()
def params(galaxy):
    return {'release': galaxy.release}
