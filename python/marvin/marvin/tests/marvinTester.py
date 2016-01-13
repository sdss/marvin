
import json, flask
from marvin import create_app
from flask.ext.testing import TestCase

try: from inspection.marvin import Inspection
except: from marvin.inspection import Inspection

class MarvinTester(TestCase):
    ''' subclass (MarvinTester.MarvinTester, TestCase), in that order '''

    def create_app(self):
        app = create_app(debug=True)
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False
        from marvin.model.database import db
        self.db = db
        return app

    def setUp(self):
        self.session = self.db.Session()
        self.longMessage = True
        self.response = None
        self.data = None
        self.insp_session = {}
        self.inspection = Inspection(self.insp_session)
        self.results = self.inspection.result()

    def tearDown(self):
        pass
        
    def _loadPage(self, type, page, params=None):
        if type == 'get':
            self.response = self.client.get(page)
        elif type == 'post':
            self.response = self.client.post(page,data=params)

        try:
            self.data = self.response.json
        except ValueError as e:
            print('Could not decode JSON: {0}'.format(e))
            self.data = None

    