#!/usr/bin/env python

import os, flask, json
import unittest, warnings
import marvinTester
from flask import url_for

class TestFeedbackPage(marvinTester.MarvinTester):

    def setUp(self):
        marvinTester.MarvinTester.setUp(self)

    def tearDown(self):
        pass

    def test_feedback_loads(self):
        with self.app.app_context():
            self._loadPage('get', url_for('feedback_page.feedback'))
            self.assertStatus(self.response, 200, 'Marvin"s feedback page should return a good status of 200')

    def _promotetracticket(self, msg=None):
        with self.app.app_context():
            self._loadPage('post', url_for('feedback_page.promotetracticket'), params={'id':1})
            self.assertEqual(self.results['status'], self.data['result']['status'],msg=msg)

    def test_promotetracticket_success(self):
        self.results['status'] = 1
        msg = 'status should be good (1)'
        self._promotetracticket(msg=msg)


if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
    
