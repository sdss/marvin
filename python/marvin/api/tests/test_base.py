#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import os
import unittest
from unittest import TestCase
from marvin.tools.core import MarvinError
from marvin.api.base import BaseView
# from marvin import config, session, datadb


class TestBase(TestCase):

    def test_reset_results(self):
        actual = BaseView()
        actual.results = {'key1': 'value1'}
        actual = self.reset_results()
        desired = {'data': None, 'status': -1, 'error': None}
        self.assertDictEqual(actual, desired)


if __name__ == '__main__':
    # set to 1 for usual '...F..' style output or to 2 for more verbose output
    verbosity = 2
    unittest.main(verbosity=verbosity)
