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
        files = ['~/tmp_testMarvinPickleSpecifyPath.pck',
                 '~/tmp_testMarvinPickleDeleteOnRestore.pck',
                 '~/tmp_testMarvinPickleOverwriteTrue.pck']
        for fn in files:
            if os.path.exists(fn):
                os.remove(fn)

    def setUp(self):
        self._files_created = []

    def tearDown(self):
        for fp in self._files_created:
            if os.path.exists(fp):
                os.remove(fp)

    def testMarvinPickleSpecifyPath(self):
        self.path_in = os.path.expanduser('~/tmp_testMarvinPickleSpecifyPath.pck')
        path_out = marvin.core.marvin_pickle.save(obj=self.data, path=self.path_in, overwrite=False)
        revivedData = marvin.core.marvin_pickle.restore(path_out)
        self._files_created.append(path_out)
        self.assertDictEqual(self.data, revivedData)

    def testMarvinPickleOverwriteTrue(self):
        fname = '~/tmp_testMarvinPickleOverwriteTrue.pck'
        self.path_in = os.path.expanduser(fname)
        open(self.path_in, 'a').close()
        path_out = marvin.core.marvin_pickle.save(obj=self.data, path=self.path_in, overwrite=True)
        self._files_created.append(path_out)
        revivedData = marvin.core.marvin_pickle.restore(path_out)
        self.assertDictEqual(self.data, revivedData)

    def testMarvinPickleDeleteOnRestore(self):
        self.path_in = os.path.expanduser('~/tmp_testMarvinPickleDeleteOnRestore.pck')
        path_out = marvin.core.marvin_pickle.save(obj=self.data, path=self.path_in, overwrite=False)
        self._files_created.append(path_out)
        revivedData = marvin.core.marvin_pickle.restore(path_out, delete=True)
        self.assertDictEqual(self.data, revivedData)
        self.assertFalse(os.path.isfile(self.path_in))

if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
