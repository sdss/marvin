#!/usr/bin/env python

import os, flask, json
import unittest, warnings
import marvinTester

class TestPlatesPage(marvinTester.MarvinTester, unittest.TestCase):

    def setUp(self):
        marvinTester.MarvinTester.setUp(self)
        self.ifuid = u'9101'
        self.plate = u'7443'
        self.cubepk = u'10'
        self.drpver = u'v1_2_0'
        self.dapver = u'v1_0_0'
        self.key = u'maps'
        self.mapid = u'kin'
        self.qatype = u'cube-none2'
        self.specpanel = u'single'
        self.issues = u'"any"'
        self.tags = []
        self.oldkey = u'maps'
        self.oldmapid = u'kin'
        self.oldqatype = u'cube-none2'
        self._buildform()


    def tearDown(self):
        pass

    def _plates_status_codes(self, page, type, params=None):
        self._loadPage(type,page, params=params)
        self.assertEqual(self.result.status_code,200, msg='Marvin"s {0} page should return a good status of 200'.format(page))

    def test_comment_addcomment_loads(self):
        self._feedback_status_codes('marvin/addcomment', 'get')
    

if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
    
