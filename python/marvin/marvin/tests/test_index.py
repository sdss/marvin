#!/usr/bin/env python

import os, flask, json
import unittest
import marvinTester
from flask import url_for

class TestIndexPage(marvinTester.MarvinTester):

    def test_index_status_code(self):
        with self.app.app_context():
            self._loadPage('get',url_for('index_page.index'))
            self.assertStatus(self.response, 200, 'Marvin"s index page should return a good status of 200')

    def test_index_data(self):
        self._loadPage('get','index/')

        with self.client.session_transaction() as sesh:
            print('session',sesh) 
        
        self.assertEqual(True,True,msg='two things should be equal')

    def test_setversion_properform(self):
        params = {'vermode':'mpl', 'version':'v1_2_0', 'dapversion':'v1_0_0', 'mplver':'MPL-2'}
        self._loadPage('post','marvin/setversion/',params=params)
        results = self.data['result']
        self.assertEqual(results['status'],1,msg='Proper form should return 1 status code')

    def test_setversion_emptyform(self):
        params = {}
        self._loadPage('post','marvin/setversion/',params=params)
        results = self.data['result']
        self.assertEqual(results['status'],-1,msg='Empty form should return -1 status code')

    def test_setversion_improperform(self):
        params = {'vermode':'mpl', 'version':'', 'dapversion':'', 'mplver':''}
        self._loadPage('post','marvin/setversion/',params=params)
        results = self.data['result']
        self.assertEqual(results['status'],-1,msg='Improper form should return -1 status code')


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
    
       