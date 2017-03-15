# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-12-08 14:24:58
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-03-15 16:09:50

from __future__ import print_function, division, absolute_import
from marvin.tests.web import MarvinWebTester


class MarvinAPITester(MarvinWebTester):

    def setUp(self):
        super(MarvinAPITester, self).setUp()
        self.blue = 'api'
        self.api_base_success = dict(status=1, error=None, traceback=None)
        self._reset_the_config()

    def _route_no_valid_params(self, url, noparam, reqtype='get', params=None, errmsg=None):
        self._load_page(reqtype, url, params=params)
        self.assert422(self.response, message='response status should be 422 for invalid params')
        self.assertIn('validation_errors', self.json.keys())
        noparam = [noparam] if not isinstance(noparam, list) else noparam
        invalid = {p: [errmsg] for p in noparam}
        self.assertDictContainsSubset(invalid, self.json['validation_errors'])

    def _assert_success(self, data, keys=None):
        self.assert200(self.response, message='response status should be 200 for ok and not {0}'.format(self.response.status_code))
        self.assertEqual(1, self.json['status'])
        self.assertDictContainsSubset(self.api_base_success, self.json)
        if isinstance(data, str):
            self.assertIn(data, self.json['data'])
        elif isinstance(data, dict):
            if keys:
                self.assertEqual(data, self.json['data'].keys())
            else:
                self.assertDictContainsSubset(data, self.json['data'])

