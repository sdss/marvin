# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-12-08 14:24:58
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-13 00:49:52

from __future__ import print_function, division, absolute_import
from marvin.tests.web import MarvinWebTester


class MarvinAPITester(MarvinWebTester):

    def setUp(self):
        super(MarvinAPITester, self).setUp()
        self.blue = 'api'
        self.api_base_success = dict(status=1, error=None, traceback=None)

    def _route_no_valid_params(self, url, noparam, reqtype='get', params=None):
        self._load_page(reqtype, url, params=params)
        self.assert422(self.response, message='response status should be 422 for invalid params')
        self.assertIn('validation_errors', self.response.json.keys())
        noparam = [noparam] if not isinstance(noparam, list) else noparam
        invalid = {p: [u'Missing data for required field.'] for p in noparam}
        self.assertDictContainsSubset(invalid, self.response.json['validation_errors'])

    def _assert_success(self, data):
        self.assert200(self.response, message='response status should be 200 for ok')
        self.assertDictContainsSubset(self.api_base_success, self.response.json)
        if isinstance(data, str):
            self.assertIn(data, self.response.json['data'])

