#!/usr/bin/env python
# encoding: utf-8
#
# test_core.py
#
# Created by Brett Andrews on 30 Nov 2016.


from __future__ import division, print_function, absolute_import

import unittest

import marvin
import marvin.tests
import marvin.core.core


class TestCoreDotable(marvin.tests.MarvinTest):
    """Test Dotable dictionary class."""

    @classmethod
    def setUpClass(cls):
        cls.normalDict = dict(A=7, b=[10, 2], C='AbCdEf', d=['ghIJ', 'Lmnop'])

    def testDotable(self):
        self.dotableDict = marvin.core.core.Dotable(self.normalDict)
        self.assertEqual(self.dotableDict['A'], self.dotableDict.A)
        self.assertListEqual(self.dotableDict['b'], self.dotableDict.b)
        self.assertEqual(self.dotableDict['C'], self.dotableDict.C)
        self.assertListEqual(self.dotableDict['d'], self.dotableDict.d)

    def testDotableCaseInsensitive(self):
        self.dotableCI = marvin.core.core.DotableCaseInsensitive(self.normalDict)
        self.assertEqual(self.dotableCI['A'], self.dotableCI.A)
        self.assertEqual(self.dotableCI['a'], self.dotableCI.a)
        self.assertEqual(self.dotableCI['A'], self.dotableCI.a)
        self.assertEqual(self.dotableCI['a'], self.dotableCI.A)
        self.assertListEqual(self.dotableCI['b'], self.dotableCI.b)
        self.assertListEqual(self.dotableCI['B'], self.dotableCI.B)
        self.assertListEqual(self.dotableCI['b'], self.dotableCI.B)
        self.assertListEqual(self.dotableCI['B'], self.dotableCI.b)
        self.assertEqual(self.dotableCI['C'], self.dotableCI.C)
        self.assertEqual(self.dotableCI['c'], self.dotableCI.c)
        self.assertEqual(self.dotableCI['C'], self.dotableCI.c)
        self.assertEqual(self.dotableCI['c'], self.dotableCI.C)
        self.assertListEqual(self.dotableCI['d'], self.dotableCI.d)
        self.assertListEqual(self.dotableCI['D'], self.dotableCI.D)
        self.assertListEqual(self.dotableCI['d'], self.dotableCI.D)
        self.assertListEqual(self.dotableCI['D'], self.dotableCI.d)

if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
