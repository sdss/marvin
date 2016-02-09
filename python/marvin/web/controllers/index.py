from flask import current_app, Blueprint, render_template
from flask.ext.classy import FlaskView, route
from marvin.tools.cube import Cube

index_page = Blueprint("index_page", __name__)

class Marvin(FlaskView):

    def index(self):
        index = {}
        index['title'] = 'Marvin'
        index['intro'] = 'Welcome to Marvin'
        index['cube'] = Cube(mangaid='12-84660')
        return render_template("index.html", **index)

    def quote(self):
        return 'getting quote'

    def get(self,id):
        return 'getting id {0}'.format(id)

    @route('/test/')
    def test(self):
        return 'new test'

Marvin.register(index_page)


