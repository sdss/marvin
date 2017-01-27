#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import unittest
from marvin import config
from marvin.api.base import BaseView


class TestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #cls.initconfig = copy.deepcopy(config)
        cls.init_sasurl = config.sasurl
        cls.init_mode = config.mode
        cls.init_urlmap = config.urlmap
        cls.init_release = config.release
        config.use_sentry = False
        config.add_github_message = False

    def setUp(self):
        # cvars = ['mode', 'drpver', 'dapver', 'mplver']
        # for var in cvars:
        #     config.__setattr__(var, self.initconfig.__getattribute__(var))
        config.sasurl = self.init_sasurl
        config.mode = self.init_mode
        config.urlmap = self.init_urlmap
        config.setMPL('MPL-4')

    def test_reset_results(self):
        bv = BaseView()
        bv.results = {'key1': 'value1'}
        bv.reset_results()
        desired = {'data': None, 'status': -1, 'error': None, 'traceback': None}
        self.assertDictEqual(bv.results, desired)

    def test_update_results(self):
        bv = BaseView()
        new_results = {'key1': 'value1'}
        bv.update_results(new_results)
        desired = {'data': None, 'status': -1, 'error': None, 'key1': 'value1', 'traceback': None}
        self.assertDictEqual(bv.results, desired)

    def test_reset_status(self):
        bv = BaseView()
        bv.results['status'] = 42
        bv.reset_status()
        self.assertEqual(bv.results['status'], -1)

    def test_add_config(self):
        bv = BaseView()
        bv.add_config()
        desired = {'data': None, 'status': -1, 'error': None, 'traceback': None,
                   'utahconfig': {'release': config.release, 'mode': config.mode}}
        self.assertDictEqual(bv.results, desired)

    def test_after_request_return_response(self):
        bv = BaseView()
        name = 'test_name'
        req = 'test_request'
        actual = bv.after_request(name, req)
        desired = 'test_request'
        self.assertEqual(actual, desired)

    def test_after_request_reset_results(self):
        bv = BaseView()
        name = 'test_name'
        req = 'test_request'
        bv.after_request(name, req)
        desired = {'data': None, 'status': -1, 'error': None, 'traceback': None}
        self.assertDictEqual(bv.results, desired)


if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
