# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-12 17:38:51
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-13 00:47:03

from __future__ import print_function, division, absolute_import
from flask_testing import TestCase
from marvin.web import create_app
from marvin import config, marvindb


class MarvinWebTester(TestCase):
    ''' Base Marvin Web Tester for Flask and API '''

    def create_app(self):
        app = create_app(debug=True, use_profiler=False)
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False
        return app

    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.session = marvindb.session
        self.long_message = True
        self.response = None
        self.data = None
        self.set_sasurl('local')
        self.urlmap = config.urlmap
        self.blue = None

    def tearDown(self):
        pass

    def set_sasurl(self, loc='local'):
        istest = True if loc == 'utah' else False
        config.switchSasUrl(loc, test=istest)

    def _load_page(self, reqtype, page, params=None):
        if reqtype == 'get':
            self.response = self.client.get(page, data=params)
        elif reqtype == 'post':
            self.response = self.client.post(page, data=params, content_type='application/x-www-form-urlencoded')

    def get_url(self, endpoint):
        return self.urlmap[self.blue][endpoint]['url']

    def assert422(self, response, message=None):
        self.assertStatus(response, 422, message)
