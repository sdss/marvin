# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-04-06 16:41:53
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-03-29 18:35:09

from __future__ import print_function, division, absolute_import
import time
import os
import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from marvin.tests.web.frontend.models import IndexPage, SearchPage


@pytest.fixture()
def page(driver, base_url):
    page = IndexPage(driver, root_uri=base_url)
    page.get('')
    return page


@pytest.mark.xfail()
@pytest.mark.timeout(45)
@pytest.mark.usefixtures('live_server')
class TestIndexPage(object):
    ''' Tests for the main Index page '''

    def test_title(self, page):
        assert 'Marvin' in page.w.title

    def test_goto_random(self, page):
        assert 'Marvin' in page.w.title
        page.imagepage.click()
        time.sleep(1)
        assert 'random' in page.w.current_url

    def test_goto_search(self, page):
        page.searchpage.click()
        time.sleep(1)
        assert 'Search' in page.w.title
        assert 'search' in page.w.current_url
        results = page.w.find_elements(By.ID, "search_results")
        assert len(results) == 0


@pytest.fixture()
def search_page(driver, base_url):
    page = SearchPage(driver, root_uri=base_url)
    page.get('search/')
    return page


@pytest.mark.xfail()
@pytest.mark.timeout(45)
@pytest.mark.usefixtures('live_server')
class TestSearchPage(object):
    ''' Tests for the main Search page '''

    def test_title(self, search_page):
        assert 'Search' in search_page.w.title

    def test_search(self, search_page):
        assert search_page.results is None
        search_page.searchbox = 'nsa.z < 0.1'
        search_page.searchbox = Keys.RETURN
        assert search_page.results is not None
        assert search_page.table is not None
