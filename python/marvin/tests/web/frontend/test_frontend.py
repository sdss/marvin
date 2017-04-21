# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-04-06 16:41:53
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-04-09 09:15:33

from __future__ import print_function, division, absolute_import
import time
import os
import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from marvin.tests.web.frontend.models import IndexPage


@pytest.fixture()
def page(driver, base_url):
    page = IndexPage(driver, root_uri=base_url)
    return page


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

    def test_goto_search(self, driver):
        driver.find_element_by_id("search_link").click()
        time.sleep(1)
        assert 'Search' in driver.title
        assert 'search' in driver.current_url
        results = driver.find_elements(By.ID, "search_results")
        assert len(results) == 0


@pytest.fixture()
def search_driver(driver, base_url):
    url = os.path.join(base_url, 'search/')
    driver.get(url)
    return driver


@pytest.mark.usefixtures('live_server')
class TestSearchPage(object):
    ''' Tests for the main Search page '''

    def test_title(self, search_driver):
        assert 'Search' in search_driver.title

    def test_search(self, search_driver):
        results = search_driver.find_elements(By.ID, "search_results")
        assert len(results) == 0
        search_driver.find_element_by_id("searchbox").send_keys('nsa.z < 0.1')
        search_driver.find_element_by_id("searchbox").send_keys(Keys.RETURN)
        results = search_driver.find_elements(By.ID, "search_results")
        assert len(results) != 0
        table = search_driver.find_elements(By.ID, "searchtablediv")
        assert len(table) != 0



