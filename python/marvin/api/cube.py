from flask.ext.classy import FlaskView, route
from flask import Blueprint
from marvin.tools.cube import Cube
import json

''' stuff that runs server-side '''

api = Blueprint("api", __name__)


class CubeView(FlaskView):
    route_base = '/cubes/'

    def index(self):
        return 'cube'

    def get(self, mangaid):
        cube = Cube(mangaid=mangaid)
        return json.dumps({mangaid: '{0},{1},{2}'.format(mangaid, cube.ra, cube.dec)})

    @route('/<mangaid>/spectra/x=<x>/y=<y>/')
    def getSpectra(self, mangaid=None, x=None, y=None):
        # Add ability to grab spectra from fits files
        cube = Cube(mangaid=mangaid)
        spectrum = cube.getSpectrum(x, y)
        return json.dumps({'data': spectrum})

CubeView.register(api)
