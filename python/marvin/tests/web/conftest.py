# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-04-28 11:34:06
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-06-28 17:13:32

from __future__ import print_function, division, absolute_import
import pytest
from marvin.web import create_app
from marvin.web.settings import TestConfig, CustomConfig
from marvin.api.api import Interaction
from marvin import marvindb, config
from flask import template_rendered, templating
from contextlib import contextmanager
import os
try:
    from urllib.parse import urlparse, urljoin
except ImportError:
    from urlparse import urlparse, urljoin


# @pytest.fixture(scope='session')
# def drpver(release):
#     drpver, dapver = config.lookUpVersions(release)
#     return drpver


# @pytest.fixture(scope='session')
# def dapver(release):
#     drpver, dapver = config.lookUpVersions(release)
#     return dapver


@pytest.fixture(scope='session')
def app():
    object_config = type('Config', (TestConfig, CustomConfig), dict())
    app = create_app(debug=True, local=True, use_profiler=False, object_config=object_config)
    return app


# def set_sasurl(loc='local', port=None):
#     if not port:
#         port = int(os.environ.get('LOCAL_MARVIN_PORT', 5000))
#     istest = True if loc == 'utah' else False
#     config.switchSasUrl(loc, test=istest, port=port)
#     response = Interaction('api/general/getroutemap', request_type='get')
#     config.urlmap = response.getRouteMap()


# @pytest.fixture()
# def saslocal():
#     set_sasurl(loc='local')


def test_db_stuff():
    assert marvindb is not None
    assert marvindb.datadb is not None
    assert marvindb.sampledb is not None
    assert marvindb.dapdb is not None
    assert 'local' == marvindb.dbtype


@pytest.fixture(scope='function')
def init_web(monkeypatch, set_config):
    config.forceDbOn()

    # monkeypath the render templating to nothing
    def _empty_render(template, context, app):
        template_rendered.send(app, template=template, context=context)
        return ""
    monkeypatch.setattr(templating, '_render', _empty_render)


@pytest.fixture(scope='function', autouse=True)
def inspection(monkeypatch):
    from brain.core.inspection import Inspection
    monkeypatch.setattr('inspection.marvin.Inspection', Inspection)


@pytest.mark.usefixtures('app, get_templates')
class Page(object):
    ''' Object representing a Web Page '''
    def __init__(self, client, blue, endpoint):
        self.app = app()
        self.url = self.get_url(blue, endpoint)
        self.json = None
        self.data = None
        self.response = None
        self.client = client

    def get_url(self, blue, endpoint):
        return config.urlmap[blue][endpoint]['url']

    def load_page(self, reqtype, page, params=None):
        if reqtype == 'get':
            self.response = self.client.get(page, query_string=params)
        elif reqtype == 'post':
            self.response = self.client.post(page, data=params, content_type='application/x-www-form-urlencoded')
        self.load_data()

    def load_data(self):
        try:
            self.json = self.response.json
        except ValueError as e:
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

    def assert_dict_contains_subset(self, first, second):
        subset = all(k in second and second[k] == v for k, v in first.items())
        assert subset is True, '{0} dictionary should be subset of {1}'.format(first, second)

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
            server_name = self.app.config.get('SERVER_NAME') or 'localhost'
            expected_location = urljoin('http://{0}'.format(server_name), location)

        valid_status_codes = (301, 302, 303, 305, 307)
        valid_status_code_str = ', '.join(str(code) for code in valid_status_codes)
        not_redirect = "HTTP Status {0} expected but got {1}".format(valid_status_code_str, self.response.status_code)
        assert self.response.status_code in valid_status_codes, message or not_redirect
        assert self.response.location == expected_location, message


@pytest.fixture()
def page(client, request, init_web):
    blue, endpoint = request.param
    page = Page(client, blue, endpoint)
    yield page


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


