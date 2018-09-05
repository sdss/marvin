# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-07 15:58:27
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-10 14:11:25

from __future__ import print_function, division, absolute_import
from marvin.api.base import BaseView
import pytest


@pytest.fixture(autouse=True)
def baseview():
    baseview = BaseView()
    yield baseview


class TestBase(object):

    def test_reset_results(self, baseview):
        baseview.results = {'key1': 'value1'}
        baseview.reset_results()
        desired = {'data': None, 'status': -1, 'error': None, 'traceback': None}
        assert baseview.results == desired, 'baseview results should be the same as desired'

    def test_update_results(self, baseview):
        new_results = {'key1': 'value1'}
        baseview.update_results(new_results)
        desired = {'data': None, 'status': -1, 'error': None, 'key1': 'value1', 'traceback': None}
        assert baseview.results == desired, 'baseview results should be the same as desired'

    def test_reset_status(self, baseview):
        baseview.results['status'] = 42
        baseview.reset_status()
        assert baseview.results['status'] == -1

    def test_add_config(self, baseview, release, mode):
        baseview.add_config()
        desired = {'data': None, 'status': -1, 'error': None, 'traceback': None,
                   'utahconfig': {'release': 'MPL-7', 'mode': 'local'}}
        assert baseview.results == desired

    def test_after_request_return_response(self, baseview):
        name = 'test_name'
        req = 'test_request'
        actual = baseview.after_request(name, req)
        desired = 'test_request'
        assert actual == desired

    def test_after_request_reset_results(self, baseview):
        name = 'test_name'
        req = 'test_request'
        baseview.after_request(name, req)
        desired = {'data': None, 'status': -1, 'error': None, 'traceback': None}
        assert baseview.results == desired

