# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-12 20:46:42
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-19 12:57:12

from __future__ import print_function, division, absolute_import
from marvin.tests.web import MarvinWebTester
import unittest


class TestIndexPage(MarvinWebTester):

    render_templates = False

    def setUp(self):
        super(TestIndexPage, self).setUp()
        self.blue = 'index_page'

    def test_assert_index_used(self):
        url = self.get_url('Marvin:index')
        self._load_page('get', url)
        self.assertEqual('', self.data)
        self.assert_template_used('index.html')


if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
