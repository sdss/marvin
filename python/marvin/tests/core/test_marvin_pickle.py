#!/usr/bin/env python
# encoding: utf-8
#
# test_marvin_pickle.py
#
# Created by Brett Andrews on 30 Nov 2016.


from __future__ import division, print_function, absolute_import

import os
import unittest

import marvin
import marvin.tests
import marvin.core.marvin_pickle


class TestMarvinPickle(marvin.tests.MarvinTest):
    """Test marvin_pickle class."""

    @classmethod
    def setUpClass(cls):
        cls.data = dict(a=7, b=[10, 2])

    def tearDown(self):
        if self.path_in in [os.path.join(os.getcwd(), it) for it in os.listdir()]:
            os.remove(self.path_in)

    def testMarvinPickleSpecifyPath(self):
        self.path_in = os.path.join(os.getcwd(), 'tmp_testMarvinPickleSpecifyPath.pck')
        path_out = marvin.core.marvin_pickle.save(obj=self.data, path=self.path_in, overwrite=False)
        revivedData = marvin.core.marvin_pickle.restore(path_out)
        self.assertDictEqual(self.data, revivedData)

    def testMarvinPickleOverwriteTrue(self):
        fname = 'tmp_testMarvinPickleOverwriteTrue.pck'
        open(fname, 'a').close()
        self.path_in = os.path.join(os.getcwd(), fname)
        path_out = marvin.core.marvin_pickle.save(obj=self.data, path=self.path_in, overwrite=True)
        revivedData = marvin.core.marvin_pickle.restore(path_out)
        self.assertDictEqual(self.data, revivedData)

    def testMarvinPickleDeleteOnRestore(self):
        self.path_in = os.path.join(os.getcwd(), 'tmp_testMarvinPickleDeleteOnRestore.pck')
        path_out = marvin.core.marvin_pickle.save(obj=self.data, path=self.path_in, overwrite=False)
        revivedData = marvin.core.marvin_pickle.restore(path_out, delete=True)
        self.assertDictEqual(self.data, revivedData)
        self.assertFalse(os.path.isfile(self.path_in))

if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
