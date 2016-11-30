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
    """Test Dotable dictionary class."""

    @classmethod
    def setUpClass(cls):
        cls.data = dict(a=7, b=[10, 2])

    def testMarvinPickleOverwriteFalse(self):
        path_in = os.path.join(os.getcwd(), 'tmp_testMarvinPickleOverwriteFalse.pck')
        path_out = marvin.core.marvin_pickle.save(obj=self.data, path=path_in, overwrite=False)
        revivedData = marvin.core.marvin_pickle.restore(path)
        self.assertDictEqual(self.data, revivedData)


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
