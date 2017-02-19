# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-12 17:38:51
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-19 13:05:01

from __future__ import print_function, division, absolute_import
from flask_testing import TestCase
from marvin.web import create_app
from marvin import config, marvindb
from marvin.tests import MarvinTest


class MarvinWebTester(MarvinTest, TestCase):
    ''' Base Marvin Web Tester for Flask and API '''

    def create_app(self):
        app = create_app(debug=True, use_profiler=False)
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False
        return app

    @classmethod
    def setUpClass(cls):
        super(MarvinWebTester, cls).setUpClass()

    def setUp(self):
        self.session = marvindb.session
        self.long_message = True
        self.response = None
        self.data = None
        self.json = None
        self.set_sasurl('local')
        self.urlmap = config.urlmap
        self.blue = None

    def tearDown(self):
        pass

    def _load_page(self, reqtype, page, params=None):
        if reqtype == 'get':
            self.response = self.client.get(page, data=params)
        elif reqtype == 'post':
            self.response = self.client.post(page, data=params, content_type='application/x-www-form-urlencoded')
        self._load_data()

    def _load_data(self):
        try:
            self.json = self.response.json
        except ValueError as e:
            self.json = None
        self.data = self.json['data'] if self.json and 'data' in self.json else ''

    def get_url(self, endpoint):
        return self.urlmap[self.blue][endpoint]['url']

    def assert422(self, response, message=None):
        self.assertStatus(response, 422, message)
