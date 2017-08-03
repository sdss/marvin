# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-04-06 15:30:50
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-08-03 16:45:46

from __future__ import print_function, division, absolute_import
import os
import pytest
import requests
from flask import url_for
from selenium import webdriver
from marvin.web import create_app
from marvin.tests.web.frontend.live_server import live_server
from marvin.web.settings import TestConfig, CustomConfig

browserstack = os.environ.get('USE_BROWSERSTACK', None)
saucelabs = os.environ.get('USE_SAUCELABS', None)

if browserstack or saucelabs:
    osdict = {'OS X': ['El Capitan', 'Sierra']}
    browserdict = {'chrome': ['55', '54'], 'firefox': ['52', '51'], 'safari': ['10', '9.0']}
else:
    osdict = {'OS X': ['El Capitan']}
    browserdict = {'chrome': ['55']}

osstuff = [(k, i) for k, v in osdict.items() for i in v]
browserstuff = [(k, i) for k, v in browserdict.items() for i in v]
osversions = {'El Capitan': '10.11', 'Sierra': '10.12'}


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
    app.config['LIVESERVER_PORT'] = 8943
    return app


@pytest.fixture(scope='function')
def base_url(live_server):
    return '{0}/marvin2/'.format(live_server.url())


@pytest.fixture(scope='function')
def driver(base_url, osinfo, browserinfo):
    ostype, os_version = osinfo
    browser, browser_version = browserinfo
    buildid = '{0}_{1}_{2}_{3}'.format(ostype.lower().replace(' ', '_'),
                                       os_version.lower().replace(' ', '_'), browser, browser_version)
    # skip some combinations
    if os_version == 'Sierra' and browser == 'safari' and browser_version == '9.0':
        pytest.skip('cannot have Sierra running safari 9.0')
    elif os_version == 'El Capitan' and browser == 'safari' and browser_version == '10':
        pytest.skip('cannot have El Capitan running safari 10')

    # set driver
    if browserstack:
        # BrowserStack
        username = os.environ.get('BROWSERSTACK_USER', None)
        access = os.environ.get('BROWSERSTACK_ACCESS_KEY', None)
        url = 'http://{0}:{1}@hub.browserstack.com:80/wd/hub'.format(username, access)
        desired_cap = {'os': ostype, 'os_version': os_version, 'browser': browser, 'build': buildid,
                       'browser_version': browser_version, 'project': 'marvin', 'resolution': '1920x1080'}
        desired_cap['browserstack.local'] = True
        desired_cap['browserstack.debug'] = True
        desired_cap['browserstack.localIdentifier'] = os.environ['BROWSERSTACK_LOCAL_IDENTIFIER']
        driver = webdriver.Remote(
            command_executor=url,
            desired_capabilities=desired_cap)
    elif saucelabs:
        # SauceLabs
        username = os.environ.get('SAUCE_USERNAME', None)
        access = os.environ.get('SAUCE_ACCESS_KEY', None)
        url = 'http://{0}:{1}@ondemand.saucelabs.com:80/wd/hub'.format(username, access)
        desired_cap = {'platform': '{0} {1}'.format(ostype, osversions[os_version]), 'browserName': browser,
                       'build': buildid, 'version': browser_version, 'screenResolution': '1920x1440',
                       'customData': {'project': 'Marvin'}, 'name': 'marvin_{0}'.format(buildid)}
        driver = webdriver.Remote(
            command_executor=url,
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




