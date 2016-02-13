from flask.ext.classy import FlaskView, route
from flask import Blueprint, url_for
from marvin.tools.cube import Cube
from marvin import config
config.drpver = 'v1_5_1'  # FIX THIS
import json
from functools import wraps

''' stuff that runs server-side '''

api = Blueprint("api", __name__)


def parseRoutePath(f):
    @wraps(f)
    def decorated_function(inst, *args, **kwargs):
        for kw in kwargs['path'].split('/'):
            if len(kw) == 0:
                continue
            var, value = kw.split('=')
            kwargs[var] = value
        kwargs.pop('path')
        return f(inst, *args, **kwargs)
    return decorated_function


class CubeView(FlaskView):
    route_base = '/cubes/'
    # decorators = [parseRoutePath]

    def index(self):
        return json.dumps({'data': 'this is a cube!'})

    @route('/<mangaid>/spectra', defaults={'path': None})
    @route('/<mangaid>/spectra/<path:path>')
    @parseRoutePath
    def getSpectra(self, mangaid=None, x=None, y=None, ra=None, dec=None, ext=None):
        # Add ability to grab spectra from fits files
        cube = Cube(mangaid=mangaid)
        result = {}
        try:
            spectrum = cube.getSpectrum(x, y)
            result['data'] = spectrum
        except Exception as e:
            result['data'] = None
            result['error'] = str(e)
            result['stuff'] = (x, y, ra, dec, ext)

        return json.dumps(result)

CubeView.register(api)
