# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-12 20:46:42
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-05-05 18:44:53

from __future__ import print_function, division, absolute_import
import pytest
from marvin.tests.web.conftest import Page, assert405, assert200, assertListIn, assert_redirects
from marvin import marvindb
from flask import url_for


@pytest.fixture()
def page(client, request):
    blue, endpoint = request.param
    page = Page(client, blue, endpoint)
    yield page


@pytest.mark.parametrize('page', [('index_page', 'Marvin:index')], ids=['index'], indirect=True)
class TestIndexPage(object):

    # def setUp(self):
    #     super(TestIndexPage, self).setUp()
    #     self.blue = 'index_page'
    #     config.setRelease('MPL-5')
    #     self.release = config.release

    # @pytest.mark.parametrize('assert_template_used', ['index.html'], indirect=True)
    # def test_assert_index_template_used(self, assert_template_used, page):
    #     #url = get_url('index_page', 'Marvin:index')
    #     print('page', page)
    #     page.load_page('get', page.url)
    #     assert '' == page.data
    #     assert_template_used('index.html')

    def test_assert_index_template_used(self, page, get_templates):
        page.load_page('get', page.url)
        assert '' == page.data
        template, context = get_templates[0]
        assert 'index.html' == template.name


@pytest.mark.parametrize('page', [('index_page', 'Marvin:database')], ids=['database'], indirect=True)
class TestDb(object):

    def test_db_works(self, page, release):
        #url = self.get_url('Marvin:database')
        page.load_page('get', page.url, params={'release': release})
        #self._load_page('get', url, params={'release': self.release})
        data = {'plate': 7443}
        page.assert_webjson_success(data)

    def test_db_post_fails(self, page, release):
        #url = self.get_url('Marvin:database')
        page.load_page('post', page.url, params={'release': release})
        #self._load_page('post', url, params={'release': self.release})
        assert405(page.response, 'allowed method should be get')


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
        #url = self.get_url('getgalidlist')
        page.load_page('post', page.url, params={'release': release})
        data = ['8485', '8485-1901', '7443', '7443-12701', '1-209232', '12-98126']
        assert200(page.response, message='response status should be 200 for ok')
        assertListIn(data, page.json)

    def test_getgalid_fail(self, page, release):
        marvindb.datadb = None
        #url = self.get_url('getgalidlist')
        page.load_page('post', page.url, params={'release': release})
        data = ['']
        assert200(page.response, message='response status should be 200 for ok')
        assert data == page.json


@pytest.mark.parametrize('page', [('index_page', 'galidselect')], ids=['galidselect'], indirect=True)
@pytest.mark.parametrize('name, id, galid', [('plate', 'plate', 8485),
                                             ('galaxy', 'plateifu', '8485-1901'),
                                             ('galaxy', 'mangaid', '1-209232'),
                                             ('main', None, 'galname')])
class TestGalIdSelect(object):

    @pytest.fixture(autouse=True)
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
        assert_redirects(page.response, redirect_url, 'page should be redirected to {0} page'.format(name))


    # def _get_galid(self, name, galid, redirect_url):
    #     data = {'galid': galid, 'release': self.release}
    #     url = self.get_url('galidselect')
    #     self._load_page('get', url, params=data)
    #     self.assert_redirects(self.response, redirect_url, 'page should be redirected to {0} page'.format(name))

    # def test_get_plate(self):
    #     self._get_galid('plate', self.plate, url_for('plate_page.Plate:get', plateid=self.plate))

    # def test_get_plateifu(self):
    #     self._get_galid('galaxy', self.plateifu, url_for('galaxy_page.Galaxy:get', galid=self.plateifu))

    # def test_get_mangaid(self):
    #     self._get_galid('galaxy', self.mangaid, url_for('galaxy_page.Galaxy:get', galid=self.mangaid))

    # def test_get_none(self):
    #     self._get_galid('main', 'galname', url_for('index_page.Marvin:index'))


# class TestGalIdSelect(TestIndexPage):

#     def _get_galid(self, name, galid, redirect_url):
#         data = {'galid': galid, 'release': self.release}
#         url = self.get_url('galidselect')
#         self._load_page('get', url, params=data)
#         self.assert_redirects(self.response, redirect_url, 'page should be redirected to {0} page'.format(name))

#     def test_get_plate(self):
#         self._get_galid('plate', self.plate, url_for('plate_page.Plate:get', plateid=self.plate))

#     def test_get_plateifu(self):
#         self._get_galid('galaxy', self.plateifu, url_for('galaxy_page.Galaxy:get', galid=self.plateifu))

#     def test_get_mangaid(self):
#         self._get_galid('galaxy', self.mangaid, url_for('galaxy_page.Galaxy:get', galid=self.mangaid))

#     def test_get_none(self):
#         self._get_galid('main', 'galname', url_for('index_page.Marvin:index'))

