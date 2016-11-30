#!/usr/bin/env python
# encoding: utf-8
#
# test_core.py
#
# Created by Brett Andrews on 30 Nov 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import unittest

import marvin
import marvin.tests
import marvin.core.core


class TestCoreDotable(marvin.tests.MarvinTest):
    """Test Dotable dictionary class."""

    @classmethod
    def setUpClass(cls):
        cls.normalDict = dict(a=7, b=[10, 2])
        cls.dotableDict = marvin.core.core.Dotable(cls.normalDict)

    def testDotable(self):
        self.assertIsEqual(self.dotableDict['a'], self.dotableDict.a)
        self.assertListEqual(self.dotableDict['b'], self.dotableDict.b)


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
