#!/usr/bin/env python

import os, flask, json
import unittest, warnings
import marvinTester
from flask import url_for

class TestPlatesPage(marvinTester.MarvinTester):

    def setUp(self):
        self.plate = 7443
        self.version = 'v1_3_3'
        self.outdir = 'stack'

    def tearDown(self):
        pass

    def test_plate_loads(self):
        with self.app.app_context():
            self._loadPage('get', url_for('plate_page.plate'))
            self.assertStatus(self.response, 200, 'Marvin"s plate page should return a good status of 200')

    # Tests for download files via rsync
    def _downloadFiles(self, params):
        with self.app.app_context():
            self._loadPage('post', url_for('plate_page.downloadFiles'), params=params)
    
    def _downloadFiles_plate(self, mode, outdir):
        params = {'id':mode, 'plate':self.plate, 'version':self.version, 'table':'null'}
        self._downloadFiles(params)
        self.assertEqual(self.data['result']['status'],1)
        self.assertEqual(self.data['result']['message'],'Success')
        if outdir == 'stack':
            self.assertIn('rsync -avz --progress --include "*{0}*fits*"'.format(mode.upper()),self.data['result']['command'])
        elif outdir =='mastar':
            self.assertIn('rsync -avz --progress --include "mastar*fits*"'.format(mode.upper()),self.data['result']['command'])

    def test_plate_cube_stack(self):
        self._downloadFiles_plate('cube', self.outdir)

    def test_plate_rss_stack(self):
        self._downloadFiles_plate('rss', self.outdir)

    def _downloadFiles_plate_fail(self, mode, outdir, msg):
        params = {'id':mode, 'plate':self.plate, 'version':self.version, 'table':'null'}
        self._downloadFiles(params)
        self.assertEqual(self.data['result']['status'],-1)
        self.assertIn(msg, self.data['result']['message'])

    def test_plate_cube_mastar_nofile(self):
        self.plate = 7999
        self.outdir = 'mastar'
        self._downloadFiles_plate_fail('cube', self.outdir, 'No such file or directory')
        
    def test_plate_cube_mastar_success(self):
        self.plate = 7999
        self.outdir = 'mastar'
        self.version='trunk'
        self._downloadFiles_plate('cube', self.outdir)

if __name__ == '__main__':
    # set to 1 for the usual '...F..' style output, or 2 for more verbose output.
    verbosity = 2
    unittest.main(verbosity=verbosity)
    
