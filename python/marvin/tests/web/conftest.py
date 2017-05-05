# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-04-28 11:34:06
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-05-05 18:38:26

from __future__ import print_function, division, absolute_import
import pytest
from marvin.web import create_app
from marvin.api.api import Interaction
from marvin import marvindb, config
from flask import template_rendered, templating, before_render_template
from contextlib import contextmanager
import os
try:
    from urllib.parse import urlparse, urljoin
except ImportError:
    from urlparse import urlparse, urljoin


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


@pytest.fixture()
def load_page(client, reqtype, page, params=None):
    if reqtype == 'get':
        response = client.get(page, query_string=params)
    elif reqtype == 'post':
        response = client.post(page, data=params, content_type='application/x-www-form-urlencoded')
    yield response


@pytest.fixture(scope='session')
def get_urlmap():
    return config.urlmap


@pytest.fixture(scope='function')
def set_sasurl(loc='local', port=None):
    if not port:
        port = int(os.environ.get('LOCAL_MARVIN_PORT', 5000))
    istest = True if loc == 'utah' else False
    config.switchSasUrl(loc, test=istest, port=port)
    response = Interaction('api/general/getroutemap', request_type='get')
    config.urlmap = response.getRouteMap()


@pytest.fixture(scope='function')
def get_url(urlmap, blue, endpoint):
    return urlmap[blue][endpoint]['url']


@pytest.fixture()
def load_data(response):
    try:
        jsondata = response.json
    except ValueError as e:
        jsondata = None
    data = jsondata['data'] if jsondata and 'data' in jsondata else ''
    yield data


def test_db_stuff():
    assert marvindb is not None
    assert marvindb.datadb is not None
    assert marvindb.sampledb is not None
    assert marvindb.dapdb is not None
    assert 'local' == marvindb.dbtype


@pytest.fixture(scope='function')
def init_web(monkeypatch, set_sasurl):
    set_sasurl('local')
    config.forceDbOn()

    # monkeypath the render templating to nothing
    def _empty_render(template, context, app):
        template_rendered.send(app, template=template, context=context)
        return ""
    monkeypatch.setattr(templating, '_rendered', _empty_render)


class Page(object):
    def __init__(self, client, blue, endpoint):
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
        assert200(self.response, message='response status should be 200 for ok')
        if isinstance(expdata, str):
            assert expdata in self.json['result']
        elif isinstance(expdata, dict):
            assert 1 == self.json['result']['status']
        elif isinstance(expdata, list):
            assertListIn(expdata, self.json['result'])


# Assert definitions from Flask-Testing
def assertListIn(a, b):
    ''' assert all items in list a are in b '''
    for item in a:
        assert item in b


def assert_status(response, status_code, message=None):
    message = message or 'HTTP Status {0} expected but got {1}'.format(status_code, response.status_code)
    assert response.status_code == status_code, message


def assert200(response, message=None):
    assert_status(response, 200, message)


def assert400(response, message=None):
    assert_status(response, 400, message)


def assert401(response, message=None):
    assert_status(response, 401, message)


def assert403(response, message=None):
    assert_status(response, 403, message)


def assert404(response, message=None):
    assert_status(response, 404, message)


def assert405(response, message=None):
    assert_status(response, 405, message)


def assert422(response, message=None):
    assert_status(response, 422, message)


def assert500(response, message=None):
    assert_status(response, 500, message)


@pytest.mark.usefixtures('app')
def assert_redirects(response, location, message=None):
    parts = urlparse(location)
    theapp = app()
    if parts.netloc:
        expected_location = location
    else:
        server_name = theapp.config.get('SERVER_NAME') or 'localhost'
        expected_location = urljoin('http://{0}'.format(server_name), location)

    valid_status_codes = (301, 302, 303, 305, 307)
    valid_status_code_str = ', '.join(str(code) for code in valid_status_codes)
    not_redirect = "HTTP Status {0} expected but got {1}".format(valid_status_code_str, response.status_code)
    assert response.status_code in valid_status_codes, message or not_redirect
    assert response.location == expected_location, message


@contextmanager
def captured_templates(app):
    recorded = []

    def record(app, template, context, **extra):
        recorded.append((template, context))

    before_render_template.connect(record)
    yield recorded
    before_render_template.disconnect(record)


@pytest.fixture()
def get_templates(app):
    with captured_templates(app) as templates:
        yield templates


