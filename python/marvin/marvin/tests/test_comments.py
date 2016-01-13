#!/usr/bin/env python

import os, flask, json
import unittest, warnings
import marvinTester
from flask import url_for

class TestCommentsPage(marvinTester.MarvinTester):

    def setUp(self):
        marvinTester.MarvinTester.setUp(self)
        self.ifuid = u'9101'
        self.plate = u'7443'
        self.cubepk = u'10'
        self.drpver = u'v1_3_3'
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
        self.oldspecpanel = u'single'
        self._buildform()

    def tearDown(self):
        marvinTester.MarvinTester.tearDown(self)

    def _comment_loads(self, page, type, params=None):
        self._loadPage(type, page, params=params)
        self.assertStatus(self.response, 200, 'Marvin"s {0} page should return a good status of 200'.format(page))

    def test_comment_addcomment_loads(self):
        with self.app.app_context():
            self._comment_loads(url_for('comment_page.addComment'), 'get')

    def test_comment_getcomment_loads(self):
        with self.app.app_context():
            self._comment_loads(url_for('comment_page.getComment'), 'get')

    def test_comment_login_loads(self):
        with self.app.app_context():
            self._comment_loads(url_for('comment_page.login'), 'get')
    
    def test_comment_getdappanel_loads(self):
        with self.app.app_context():
            self._comment_loads(url_for('comment_page.getdappanel'), 'post', params=self.dapform)
            self._dappanel_status_asserts_ok()

    def test_comment_getdapspeclist_loads(self):
        with self.app.app_context():
            self._comment_loads(url_for('comment_page.getdapspeclist'), 'post', params=self.dapform)

    def _buildform(self):
        self.dapform = {'ifu': self.ifuid, 'cubepk': self.cubepk, 'plateid': self.plate, 'drpver': self.drpver, 'dapver': self.dapver,
        'key': self.key, 'mapid': self.mapid, 'qatype': self.qatype, 'issues': self.issues, 'tags': self.tags, 'specpanel': self.specpanel,
        'oldkey': self.oldkey, 'oldmapid': self.oldmapid, 'oldqatype': self.oldqatype, 'oldspecpanel':self.oldspecpanel,  
        'dapqa_comment1_1': u'', 'dapqa_comment1_2': u'', 'dapqa_comment1_3': u'', 'dapqa_comment1_4': u'', 'dapqa_comment1_5': u'', 
        'dapqa_comment1_6': u'', 'dapqa_comment2_1': u'', 'dapqa_comment2_2': u'', 'dapqa_comment2_3': u'', 'dapqa_comment2_4': u'', 
        'dapqa_comment2_5': u'', 'dapqa_comment2_6': u'', 'dapqa_comment3_1': u'', 'dapqa_comment3_2': u'', 'dapqa_comment3_3': u'', 
        'dapqa_comment3_4': u'', 'dapqa_comment3_5': u'', 'dapqa_comment3_6': u''}
    
    def _loadCrashForm(self, name):
        form = {}
        with open(name) as f:
            for line in f:
                (key, val) = line.split()
                form[key] = val
        self.dapform = form

    def _changeform(self,dict):
        for key,value in dict.iteritems():
            self.dapform[key] = value

    def _dappanel_status_asserts_ok(self):
        self.assertEqual(self.data['result']['status'], 1, msg='main dappanel should have status of 1')
        self.assertEqual(self.data['result']['setsession']['status'], 1, msg='main dappanel setsession should have status of 1')
        self.assertEqual(self.data['result']['getsession']['status'], 1, msg='main dappanel getsession should have status of 1')
    
    def _dappanel_message_asserts(self):
        self.assertIn('message',self.data['result']['panelmsg'])

    def _dappanel_formfail(self, msg):
        with self.app.app_context():
            self._loadPage('post', url_for('comment_page.getdappanel'), params=self.dapform)
            self.assertEqual(True, msg)        

    def test_dappanel_formfail_oldqatype(self):
        self._changeform({'oldqatype':'badqatype'})
        with self.app.app_context():
            self._loadPage('post', url_for('comment_page.getdappanel'), params=self.dapform)
            errmsg = 'Error in setSessionDAPComments: Error splitting old qatype' 
            self.assertEqual(self.data['result']['setsession']['status'],-1, msg='Marvin should die in setSessionDAPComments, status=-1')
            self.assertIn(errmsg, self.data['result']['setsession']['message'], msg='Error message should be value unpack')

    def test_dappanel_formfail_qatype(self):
        self._changeform({'qatype':' '})        
        errMsg = 'Error splitting qatype {0}'.format(self.dapform['qatype'])
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            with self.app.app_context():
                self._loadPage('post', url_for('comment_page.getdappanel'), params=self.dapform)
        self.assertIs(cm[-1].category,RuntimeWarning)
        self.assertIn(errMsg,str(cm[-1].message))  

    def test_crashform1_oldqatype(self):
        self._loadCrashForm('data/crashform_1.txt')
        with self.app.app_context():
            self._comment_loads(url_for('comment_page.getdappanel'), 'post', params=self.dapform) 
            self.assertEqual(self.data['result']['status'],-1)
            self.assertIn('Error splitting old qatype',self.data['result']['panelmsg'])

if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
    
       