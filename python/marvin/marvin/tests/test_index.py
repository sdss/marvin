#!/usr/bin/env python

import os, flask, json
import unittest
import marvinTester

class TestIndexPage(marvinTester.MarvinTester, unittest.TestCase):

    def _loadPage(self, type, page, params=None):
        if type == 'get':
            self.result = self.app.get(page)
        elif type == 'post':
            self.result = self.app.post(page,data=params)
    
    def test_index_status_code(self):
        self._loadPage('get','marvin/index')
        self.assertEqual(self.result.status_code,200, msg='Marvin"s index page should return a good status of 200')

    def test_index_data(self):
        self._loadPage('get','marvin/index')

        with self.app.session_transaction() as sesh:
            print('session',sesh) 
        
        self.assertEqual(True,False,msg='two things should be equal')

    def test_setversion_properform(self):
        params = {'vermode':'mpl', 'version':'v1_2_0', 'dapversion':'v1_0_0', 'mplver':'MPL-2'}
        self._loadPage('post','marvin/setversion',params=params)
        results = (json.loads(self.result.data))['result']
        self.assertEqual(results['status'],1,msg='Proper form should return 1 status code')

    def test_setversion_emptyform(self):
        params = {}
        self._loadPage('post','marvin/setversion',params=params)
        results = (json.loads(self.result.data))['result']
        self.assertEqual(results['status'],-1,msg='Empty form should return -1 status code')

    def test_setversion_improperform(self):
        params = {'vermode':'mpl', 'version':'', 'dapversion':'', 'mplver':''}
        self._loadPage('post','marvin/setversion',params=params)
        results = (json.loads(self.result.data))['result']
        self.assertEqual(results['status'],-1,msg='Improper form should return -1 status code')


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
    
       