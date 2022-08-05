# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-12 20:46:42
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-06-06 14:38:39

from __future__ import print_function, division, absolute_import
import pytest
from marvin import marvindb
from flask import url_for

pytestmark = pytest.mark.uses_web


@pytest.mark.parametrize('page', [('index_page', 'Marvin:index')], ids=['index'], indirect=True)
class TestIndexPage(object):

    def test_assert_index_template_used(self, page, get_templates):
        page.load_page('get', page.url)
        assert '' == page.data
        template, context = get_templates[0]
        assert 'index.html' == template.name


@pytest.mark.parametrize('page', [('index_page', 'Marvin:database')], ids=['database'], indirect=True)
class TestDb(object):

    def test_db_works(self, page, release):
        page.load_page('get', page.url, params={'release': release})
        data = {'plate': 7443}
        page.assert_webjson_success(data)

    def test_db_post_fails(self, page, release):
        page.load_page('post', page.url, params={'release': release})
        page.assert405('allowed method should be get')


@pytest.mark.parametrize('page', [('index_page', 'selectmpl')], ids=['selectmpl'], indirect=True)
class TestSelectMPL(object):

    def test_select_mpl(self, page, release, drpver, dapver):
        page.load_page('post', page.url, params={'release': release})
        data = {'current_release': release, 'current_drpver': drpver, 'current_dapver': dapver}
        page.assert_webjson_success(data)
        self._release_in_session(page, data)

    def _release_in_session(self, page, data):
        with page.client.session_transaction() as sess:
            sess['release'] = data['current_release']
            sess['drpver'] = data['current_drpver']
            sess['dapver'] = data['current_dapver']


@pytest.mark.parametrize('page', [('index_page', 'getgalidlist')], ids=['getgalid'], indirect=True)
class TestGetGalIdList(object):

    def test_getgalid_success(self, page, release):
        page.load_page('post', page.url, params={'release': release})
        data = ['8485', '8485-1901', '1-209232']
        page.assert200(message='response status should be 200 for ok')
        page.assertListIn(data, page.json)

    def test_getgalid_fail(self, page, release):
        marvindb.datadb = None
        page.load_page('post', page.url, params={'release': release})
        data = ['']
        page.assert200(message='response status should be 200 for ok')
        assert data == page.json


@pytest.mark.parametrize('page', [('index_page', 'galidselect')], ids=['galidselect'], indirect=True)
@pytest.mark.parametrize('name, id, galid', [('plate', 'plate', 8485),
                                             ('galaxy', 'plateifu', '8485-1901'),
                                             ('galaxy', 'mangaid', '1-209232'),
                                             ('main', None, None)])
class TestGalIdSelect(object):

    def get_url(self, name, galid):
        if name == 'plate':
            return url_for('plate_page.Plate:get', plateid=galid)
        elif name == 'plateifu':
            return url_for('galaxy_page.Galaxy:get', galid=galid)
        elif name == 'mangaid':
            return url_for('galaxy_page.Galaxy:get', galid=galid)
        elif name is None:
            return url_for('index_page.Marvin:index')

    def test_get_galid(self, page, release, name, id, galid):
        data = {'galid': galid, 'release': release}
        page.load_page('get', page.url, params=data)
        redirect_url = self.get_url(id, galid)
        if id:
            page.assert_redirects(redirect_url, 'page should be redirected to {0} page'.format(name))
        else:
            page.assert422(message='response should be 422 for no name input')


@pytest.mark.xfail(reason='until we can deal with Credentials')
@pytest.mark.parametrize('page', [('index_page', 'login', False)], ids=['login'], indirect=True)
@pytest.mark.parametrize('data, exp',
                         [({'username': '', 'password': ''}, {'ready': False, 'status': -1, 'message': 'Login  is not valid!'}),
                          ({'username': 'sdss', 'password': 'password'}, {'ready': False, 'status': -1, 'message': 'Login sdss is not valid!'}),
                          ({'username': 'bac29', 'password': 'password'}, {'ready': False, 'status': -1, 'message': 'Login bac29 is not valid!'}),
                          ({'username': 'test', 'password': 'test'}, {'ready': True, 'status': 1, 'message': 'Login Successful!', 'membername': 'SDSS User'})],
                         ids=['no_input', 'wrong_pass', 'wrong_user', 'success'])
class TestLogin(object):

    def test_login(self, inspection, page, release, data, exp):
        data['release'] = release
        page.load_page('post', page.url, params=data)
        page.assert200('response status should be 200 for ok')
        assert exp['status'] == page.json['result']['status']
        assert exp['message'] == page.json['result']['message']
        if 'membername' in exp and 'membername' in page.json['result']:
            assert exp['membername'] == page.json['result']['membername']

