# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-04-08 14:59:40
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-04-09 09:07:34

from __future__ import print_function, division, absolute_import
from page_objects import PageObject, PageElement


class IndexPage(PageObject):
    imagepage = PageElement(id_='image_link')
    searchpage = PageElement(id_='search_link')


class SearchPage(PageObject):
    searchbox = PageElement(id_='searchbox')
    results = PageElement(id_='search_results')
    table = PageElement(id_='searchtablediv')

