from flask.ext.classy import FlaskView, route
from flask import Blueprint, url_for, current_app
from marvin.tools.cube import Cube
from marvin.tools.core import MarvinError
import json
import urllib
from functools import wraps
from marvin import config
config.drpver = 'v1_5_1'  # FIX THIS

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


def parseName(name):
    try:
        namesplit = name.split('-')
    except AttributeError as e:
        raise AttributeError('Could not split on input name {0}: {1}'.format(name, e))

    if len(namesplit) == 1:
        raise MarvinError('Input name not of type manga-id or plate-ifu')
    else:
        if len(namesplit[0]) == 4:
            plateifu = name
            mangaid = None
        else:
            mangaid = name
            plateifu = None

    return mangaid, plateifu

# JSON results to return from API
result = {'data': None, 'status': -1, 'error': None}


class CubeView(FlaskView):
    route_base = '/cubes/'
    # decorators = [parseRoutePath]

    def index(self):
        # return json.dumps({'data': 'this is a cube!'})
        func_list = {}
        output = []
        for rule in current_app.url_map.iter_rules():
            options = {}
            for arg in rule.arguments:
                options[arg] = "[{0}]".format(arg)

            methods = ','.join(rule.methods)
            url = url_for(rule.endpoint, **options)
            line = urllib.unquote("{:50s} {:20s} {}".format(rule.endpoint, methods, url))
            output.append(line)

        res = {'data': []}
        for line in sorted(output):
            res['data'].append(line)

        return json.dumps(res)

    def _getCube(self, name):
        cube = None

        # parse name into either mangaid or plateifu
        try:
            mangaid, plateifu = parseName(name)
        except Exception as e:
            result['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(e))
            return cube

        try:
            cube = Cube(mangaid=mangaid, plateifu=plateifu, mode='local')
            result['status'] = 1
        except Exception as e:
            result['error'] = 'Failed to retrieve cube {0}: {1}'.format(name, str(e))

        return cube

    def get(self, name):

        cube = self._getCube(name)
        if cube:
            result['data'] = {name: '{0},{1},{2}'.format(name, cube.plate)}
        return json.dumps(result)

    @route('/<name>/spectra', defaults={'path': None})
    @route('/<name>/spectra/<path:path>', endpoint='getspectra')
    @parseRoutePath
    def getSpectra(self, name=None, x=None, y=None, ra=None, dec=None, ext=None):
        # Add ability to grab spectra from fits files

        cube = self._getCube(name)
        if not cube:
            result['error'] = 'getSpectra: No cube: {0}'.format(result['error'])
            return json.dumps(result)

        try:
            spectrum = cube.getSpectrum(x=x, y=y, ra=ra, dec=dec, ext=ext)
            result['data'] = spectrum
            result['status'] = 1
        except Exception as e:
            result['error'] = 'getSpectra: Failed to get spectrum: {0}'.format(str(e))
            result['stuff'] = (name, x, y, ra, dec, ext, plateifu, mangaid)

        return json.dumps(result)

CubeView.register(api)
