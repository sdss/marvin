
from marvin import create_app
app = create_app(debug=True)
app.config['TESTING'] = True
app.config['WTF_CSRF_ENABLED'] = False

from marvin.model.database import db

class MarvinTester(object):
    """subclass (MarvinTester.MarvinTester, unittest.TestCase), in that order."""
    def setUp(self):
        self.app = app.test_client()
        # self.app = app.test_client()
        self.session = db.Session()
        self.longMessage=True


    def _loadPage(self, type, page, params=None):
        if type == 'get':
            self.result = self.app.get(page)
        elif type == 'post':
            self.result = self.app.post(page,data=params)
        self.data = json.loads(self.result.data)

    