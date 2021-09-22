# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-04-06 15:30:50
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-04-03 16:08:48

from __future__ import print_function, division, absolute_import
import os
import pytest
import requests
from flask import url_for
from selenium import webdriver
from marvin.web import create_app
from tests.web.frontend.live_server import live_server
from marvin.web.settings import TestConfig, CustomConfig

browserstack = os.environ.get('USE_BROWSERSTACK', None)

if browserstack:
    osdict = {'OS X': ['El Capitan', 'Sierra']}
    browserdict = {'chrome': ['55', '54'], 'firefox': ['52', '51'], 'safari': ['10', '9.1']}
else:
    osdict = {'OS X': ['El Capitan']}
    browserdict = {'chrome': ['55']}


osstuff = [(k, i) for k, v in osdict.items() for i in v]
browserstuff = [(k, i) for k, v in browserdict.items() for i in v]


@pytest.fixture(params=osstuff)
def osinfo(request):
    return request.param


@pytest.fixture(params=browserstuff)
def browserinfo(request):
    return request.param


@pytest.fixture(scope='session')
def app():
    object_config = type('Config', (TestConfig, CustomConfig), dict())
    app = create_app(debug=True, local=True, object_config=object_config)
    # app = create_app(debug=True, local=True, use_profiler=False)
    # app.config['TESTING'] = True
    # app.config['WTF_CSRF_ENABLED'] = False
    # app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False
    app.config['LIVESERVER_PORT'] = 8443
    return app


@pytest.fixture(scope='function')
def base_url(live_server):
    url = live_server.url()
    #url = url.replace('localhost', '127.0.0.1')
    return '{0}/marvin/'.format(url)


@pytest.fixture(scope='function')
def driver(base_url, osinfo, browserinfo):
    ostype, os_version = osinfo
    browser, browser_version = browserinfo
    buildid = '{0}_{1}_{2}_{3}'.format(ostype.lower().replace(' ', '_'),
                                       os_version.lower().replace(' ', '_'), browser, browser_version)
    # skip some combinations
    if os_version == 'Sierra' and browser == 'safari' and browser_version == '9.1':
        pytest.skip('cannot have Sierra running safari 9.1')
    elif os_version == 'El Capitan' and browser == 'safari' and browser_version == '10':
        pytest.skip('cannot have El Capitan running safari 10')

    # set driver
    if browserstack:
        username = os.environ.get('BROWSERSTACK_USER', None)
        access = os.environ.get('BROWSERSTACK_ACCESS_KEY', None)
        desired_cap = {'os': ostype, 'os_version': os_version, 'browser': browser, 'build': buildid,
                       'browser_version': browser_version, 'project': 'marvin', 'resolution': '1920x1080'}
        desired_cap['browserstack.local'] = True
        desired_cap['browserstack.debug'] = True
        desired_cap['browserstack.localIdentifier'] = os.environ['BROWSERSTACK_LOCAL_IDENTIFIER']
        driver = webdriver.Remote(
            command_executor='http://{0}:{1}@hub.browserstack.com:80/wd/hub'.format(username, access),
            desired_capabilities=desired_cap)
    else:
        if browser == 'chrome':
            driver = webdriver.Chrome()
        elif browser == 'firefox':
            driver = webdriver.Firefox()
        elif browser == 'safari':
            driver = webdriver.Safari()

    driver.get(base_url)
    yield driver
    # teardown
    driver.quit()


@pytest.mark.usefixtures('live_server')
class TestLiveServer(object):

    def test_server_is_up_and_running(self, base_url):
        response = requests.get(base_url)
        assert response.status_code == 200




