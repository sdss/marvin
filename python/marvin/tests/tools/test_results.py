#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import unittest
from marvin import config
# from marvin.tools.query.query import Query
from marvin.tools.query.results import Results


class TestQuery(unittest.TestCase):

    """
    @classmethod
    def setUpClass(self):
        test_params = {'k{}'.format(i): 'v{}'.format(i) for i in range(5)}
        self.query = Query()
        self.query.set_params(test_params)
        self.results = Reults(self.query)
    """

    def test_sort(self):
        self.results
        self.results.sort()
        self.assertDictEqual(self.results.data(), desired)

if __name__ == '__main__':
    verbosity = 2
    unittest.main(verbosity=verbosity)
