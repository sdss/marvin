#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import unittest
# from marvin import config
from marvin.tools.query.query import Query


class TestQuery(unittest.TestCase):

    """
    @classmethod
    def setUpClass(self):
        self.qry = Query()
        self.mode = 'test'
    """

    def test_reset_params(self):
        qry = Query()
        qry.params = {'key1': 'value1'}
        qry.reset_params()
        self.assertDictEqual(qry.params(), {})

    def test_mplver(self):
        qry = Query()
        self.assertEqual(qry.mode, 'MPL-42')

if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
