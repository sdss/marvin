#!/usr/bin/env python

import os, flask, json
import unittest, warnings
import marvinTester

class TestCommentsPage(marvinTester.MarvinTester, unittest.TestCase):

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

    def _loadPage(self, type, page, params=None):
        if type == 'get':
            self.result = self.app.get(page)
        elif type == 'post':
            self.result = self.app.post(page,data=params)
        self.data = json.loads(self.result.data)
    
    def _comment_status_codes(self, page, type, params=None):
        self._loadPage(type,page, params=params)
        self.assertEqual(self.result.status_code,200, msg='Marvin"s {0} page should return a good status of 200'.format(page))

    def test_comment_addcomment_loads(self):
        self._comment_status_codes('marvin/addcomment', 'get')

    def test_comment_getcomment_loads(self):
        self._comment_status_codes('marvin/getcomment', 'get')

    def test_comment_login_loads(self):
        self._comment_status_codes('marvin/login', 'get')

    def test_comment_getdappanel_loads(self):
        self._comment_status_codes('marvin/getdappanel', 'post', params=self.dapform)
        self._dappanel_status_asserts()

    def test_comment_getdapspeclist_loads(self):
        self._comment_status_codes('marvin/getdapspeclist', 'post', params=self.dapform)

    def _buildform(self):
        self.dapform = {'ifu': self.ifuid, 'cubepk': self.cubepk, 'plateid': self.plate, 'drpver': self.drpver, 'dapver': self.dapver,
        'key': self.key, 'mapid': self.mapid, 'qatype': self.qatype, 'issues': self.issues, 'tags': self.tags, 'specpanel': self.specpanel,
        'oldkey': self.oldkey, 'oldmapid': self.oldmapid, 'oldqatype': self.oldqatype,  
        'dapqa_comment1_1': u'', 'dapqa_comment1_2': u'', 'dapqa_comment1_3': u'', 'dapqa_comment1_4': u'', 'dapqa_comment1_5': u'', 
        'dapqa_comment1_6': u'', 'dapqa_comment2_1': u'', 'dapqa_comment2_2': u'', 'dapqa_comment2_3': u'', 'dapqa_comment2_4': u'', 
        'dapqa_comment2_5': u'', 'dapqa_comment2_6': u'', 'dapqa_comment3_1': u'', 'dapqa_comment3_2': u'', 'dapqa_comment3_3': u'', 
        'dapqa_comment3_4': u'', 'dapqa_comment3_5': u'', 'dapqa_comment3_6': u''}

    '''
    ('first dapform', {'cubepk': u'10', 'drpver': u'v1_2_0', 'issues': u'"any"', 'oldkey': u'', 'mapid': u'kin', 'specpanel': u'single', 
        'oldqatype': u'', 'dapver': u'v1_0_0', 'tags': [], 'plateid': u'7443', 'key': u'maps', 'ifu': u'9101', 'qatype': u'cube-none2', 'oldmapid': u'',
        'dapqa_comment1_1': [u'', u''], 'dapqa_comment1_3': u'', 'dapqa_comment1_2': u'', 'dapqa_comment1_5': u'', 'dapqa_comment1_4': u'', 
        'dapqa_comment1_6': u'',  'dapqa_comment2_6': u'', 'dapqa_comment2_4': u'', 'dapqa_comment2_5': u'', 'dapqa_comment2_2': u'', 
        'dapqa_comment2_3': u'', 'dapqa_comment2_1': u'', 'dapqa_comment3_6': u'', 'dapqa_comment3_5': u'', 'dapqa_comment3_4': u'', 
        'dapqa_comment3_3': u'', 'dapqa_comment3_2': u'', 'dapqa_comment3_1': [u'', u'']})
    '''

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

    def _dappanel_status_asserts(self):
        self.assertEqual(self.data['result']['status'], 1, msg='main dappanel should have status of 1')
        self.assertEqual(self.data['result']['setsession']['status'], 1, msg='main dappanel setsession should have status of 1')
        self.assertEqual(self.data['result']['getsession']['status'], 1, msg='main dappanel getsession should have status of 1')

    def _dappanel_formfail(self, msg):
        self._loadPage('post', 'marvin/getdappanel', params=self.dapform)
        self.assertEqual(True, msg)        

    def test_dappanel_formfail_oldqatype(self):
        self._changeform({'oldqatype':'badqatype'})
        self._loadPage('post', 'marvin/getdappanel', params=self.dapform)
        errmsg = 'Error in setSessionDAPComments: Error splitting old qatype' 
        self.assertEqual(self.data['result']['setsession']['status'],-1, msg='Marvin should die in setSessionDAPComments, status=-1')
        self.assertIn(errmsg, self.data['result']['setsession']['message'], msg='Error message should be value unpack')

    def test_dappanel_formfail_qatype(self):
        self._changeform({'qatype':' '})        
        errMsg = 'Error splitting qatype {0}'.format(self.dapform['qatype'])
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            self._loadPage('post', 'marvin/getdappanel', params=self.dapform)
        self.assertIs(cm[-1].category,RuntimeWarning)
        self.assertIn(errMsg,str(cm[-1].message))  

    def test_crashform1(self):
        self._loadCrashForm('data/crashform_1.txt')
        self._comment_status_codes('marvin/getdappanel', 'post', params=self.dapform) 
        self._dappanel_status_asserts()


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
    
       