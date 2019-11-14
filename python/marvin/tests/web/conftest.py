# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-04-28 11:34:06
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-13 13:59:20

from __future__ import print_function, division, absolute_import
import pytest
from marvin.web import create_app
from marvin.web.settings import TestConfig, CustomConfig
from marvin import marvindb, config
from marvin.web.extensions import limiter
from flask import template_rendered, templating
from contextlib import contextmanager
import numpy as np
#from pytest_flask.fixtures import config as app_config

try:
    from urllib.parse import urlparse, urljoin
except ImportError:
    from urlparse import urlparse, urljoin


@pytest.fixture(scope='session')
def app():
    object_config = type('Config', (TestConfig, CustomConfig), dict())
    app = create_app(debug=True, local=True, object_config=object_config)
    limiter.enabled = False
    return app


def test_db_stuff():
    assert marvindb is not None
    assert marvindb.datadb is not None
    assert marvindb.sampledb is not None
    assert marvindb.dapdb is not None
    assert 'local' == marvindb.dbtype


@pytest.fixture(scope='function')
def init_web(monkeypatch, monkeyauth, set_config):
    config.forceDbOn()
    config.login()

    # monkeypath the render templating to nothing
    def _empty_render(template, context, app):
        template_rendered.send(app, template=template, context=context)
        return ""
    monkeypatch.setattr(templating, '_render', _empty_render)


@pytest.fixture(scope='function', autouse=True)
def inspection(monkeypatch):
    from brain.core.inspection import Inspection
    try:
        monkeypatch.setattr('inspection.marvin.Inspection', Inspection)
    except Exception:
        pass


@pytest.mark.usefixtures('app, get_templates')
class Page(object):
    ''' Object representing a Web Page '''
    def __init__(self, app, client, blue, endpoint):
        self.config = app.config
        self.url = self.get_url(blue, endpoint)
        self.json = None
        self.data = None
        self.response = None
        self.client = client

    def get_url(self, blue, endpoint):
        return config.urlmap[blue][endpoint]['url']

    def load_page(self, reqtype, page, params=None):
        headers = {'Authorization': 'Bearer {0}'.format(config.token)}
        if reqtype == 'get':
            self.response = self.client.get(page, query_string=params, headers=headers)
        elif reqtype == 'post':
            self.response = self.client.post(page, data=params, headers=headers,
                                             content_type='application/x-www-form-urlencoded')
        self.load_data()

    def load_data(self):
        try:
            self.json = self.response.json
        except ValueError:
            self.json = None
        self.data = self.json['data'] if self.json and 'data' in self.json else ''

    def assert_webjson_success(self, expdata):
        self.assert200(message='response status should be 200 for ok')
        if isinstance(expdata, str):
            assert expdata in self.json['result']
        elif isinstance(expdata, dict):
            assert self.json['result']['status'] == 1
        elif isinstance(expdata, list):
            self.assertListIn(expdata, self.json['result'])

    def route_no_valid_webparams(self, template, context, noparam, reqtype='get', params=None, errmsg=None):
        self.assert422(message='response status should be 422 for invalid params')
        assert 'errors/unprocessable_entity.html' == template.name, 'template name should be unprocessable_entity'
        noparam = [noparam] if not isinstance(noparam, list) else noparam
        invalid = {p: [errmsg] for p in noparam}
        assert context['data'] == invalid, 'response should contain validation error dictionary'

    # Assert definitions from Flask-Testing
    def assertListIn(self, a, b):
        ''' assert all items in list a are in b '''
        for item in a:
            assert item in b

    @staticmethod
    def _compare_values_is_subset(aa, bb):
        """Checks if one value or list is a subset of other."""

        if not hasattr(aa, '__iter__') and not hasattr(aa, '__getitem__'):
            if aa != bb and not np.isclose(aa, bb):
                return False
        else:
            # Checks whether the elements are a list of lists. If so, recursively calls itself.
            try:
                if not set(aa).issubset(set(bb)):
                    return False
            except Exception:
                if len(aa) > len(bb):
                    return False
                else:
                    for ii in range(len(aa)):
                        return Page._compare_values_is_subset(aa[ii], bb[ii])

        return True

    def assert_dict_contains_subset(self, subset, dictionary):
        """Asserts whether a dictionary is a subset of other."""

        missing = []
        mismatched = []
        for key, value in subset.items():
            if key not in dictionary:
                missing.append(key)
            elif not self._compare_values_is_subset(value, dictionary[key]):
                mismatched.append((key, (value, dictionary[key])))

        assert not (missing or mismatched), \
            '{0} dictionary should be subset of {1}'.format(subset, dictionary)

    def assert_status(self, status_code, message=None):
        message = message or 'HTTP Status {0} expected but got {1}'.format(status_code, self.response.status_code)
        assert self.response.status_code == status_code, message

    def assert200(self, message=None):
        self.assert_status(200, message)

    def assert400(self, message=None):
        self.assert_status(400, message)

    def assert401(self, message=None):
        self.assert_status(401, message)

    def assert403(self, message=None):
        self.assert_status(403, message)

    def assert404(self, message=None):
        self.assert_status(404, message)

    def assert405(self, message=None):
        self.assert_status(405, message)

    def assert422(self, message=None):
        self.assert_status(422, message)

    def assert500(self, message=None):
        self.assert_status(500, message)

    def assert_redirects(self, location, message=None):
        parts = urlparse(location)
        if parts.netloc:
            expected_location = location
        else:
            server_name = self.config.get('SERVER_NAME') or 'localhost'
            expected_location = urljoin('http://{0}'.format(server_name), location)

        valid_status_codes = (301, 302, 303, 305, 307)
        valid_status_code_str = ', '.join(str(code) for code in valid_status_codes)
        not_redirect = "HTTP Status {0} expected but got {1}".format(valid_status_code_str, self.response.status_code)
        assert self.response.status_code in valid_status_codes, message or not_redirect
        assert self.response.location == expected_location, message


def _split_request(data):
    ''' splits a page request params to check for login boolean '''
    if len(data) == 2:
        login = True
        blue, endpoint = data
    else:
        blue, endpoint, login = data
    return blue, endpoint, login


@pytest.fixture()
def page(user, client, request, init_web, app):
    ''' general page fixture to use for all web tests '''
    blue, endpoint, dologin = _split_request(request.param)
    page = Page(app, client, blue, endpoint)
    print('login config', config.sasurl, dologin)
    if dologin:
        login(page)
    yield page
    url = page.get_url('index_page', 'Marvin:clear_session')
    page.load_page('get', url)


def login(page):
    ''' perform a web login '''
    data = {'username': 'test', 'password': 'test', 'release': 'MPL-6'}
    url = page.get_url('index_page', 'login')
    page.load_page('post', url, params=data)


@contextmanager
def captured_templates(app):
    ''' Records which templates are used '''
    recorded = []

    def record(app, template, context, **extra):
        recorded.append((template, context))

    template_rendered.connect(record)
    yield recorded
    template_rendered.disconnect(record)


@pytest.fixture()
def get_templates(app):
    ''' Fixture that returns which jinja template used '''
    with captured_templates(app) as templates:
        yield templates
