from flask import current_app, Blueprint, render_template
from flask.ext.classy import FlaskView, route
from marvin.tools.cube import Cube

index = Blueprint("index_page", __name__)

class Marvin(FlaskView):

    def index(self):
        index = {}
        index['title'] = 'Marvin'
        index['intro'] = 'Welcome to Marvin'
        cube = Cube(mangaid='12-193534')
        index['cube'] = cube
        index['spectra'] = cube.getSpectrum(1,2)
        return render_template("index.html", **index)

    def quote(self):
        return 'getting quote'

    def get(self,id):
        return 'getting id {0}'.format(id)

    @route('/test/')
    def test(self):
        return 'new test'

Marvin.register(index)

