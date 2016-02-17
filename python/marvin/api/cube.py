from flask.ext.classy import FlaskView, route
from flask import Blueprint, url_for, current_app
from marvin.tools.cube import Cube
import json
import urllib
from marvin import config
from marvin.utils.general import parseRoutePath, parseName

config.drpver = 'v1_5_1'  # FIX THIS

''' stuff that runs server-side '''

api = Blueprint("api", __name__)


class Results(object):
    '''Container of results to return from API as JSON.'''

    def __init__:
        self.reset()

    def reset(self):
        self.output = {'data': None, 'status': -1, 'error': None}


def _getCube(name):
    ''' Retrieve a cube using marvin tools '''

    cube = None
    results = Results()

    # parse name into either mangaid or plateifu
    try:
        mangaid, plateifu = parseName(name)
    except Exception as e:
        results.output['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(e))
        return cube, results

    try:
        cube = Cube(mangaid=mangaid, plateifu=plateifu, mode='local')
        results.output['status'] = 1
    except Exception as e:
        results.output['error'] = 'Failed to retrieve cube {0}: {1}'.format(name, str(e))

    return cube, results


class CubeView(FlaskView):
    ''' Class describing API calls related to MaNGA Cubes '''

    route_base = '/cubes/'
    # decorators = [parseRoutePath]

    def index(self):
        results = Results()
        return json.dumps({'data': 'this is a cube!'})
        '''
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
        '''

    def get(self, name):
        results = Results()
        cube, res = _getCube(name)
        if cube:
            results.output['data'] = {name: '{0},{1},{2},{3}'.format(name, cube.plate, cube.ra, cube.dec)}
        out = json.dumps(results.output)
        results.reset()
        return out

    @route('/<name>/spectra', defaults={'path': None})
    @route('/<name>/spectra/<path:path>', endpoint='getspectra')
    @parseRoutePath
    def getSpectra(self, name=None, x=None, y=None, ra=None, dec=None, ext=None):
        # Add ability to grab spectra from fits files
        cube, results = _getCube(name)
        if not cube:
            results = Results()
            results.output['error'] = 'getSpectra: No cube: {0}'.format(result['error'])
            out = json.dumps(results.output)
            results.reset()
            return out

        try:
            spectrum = cube.getSpectrum(x=x, y=y, ra=ra, dec=dec, ext=ext)
            results.output['data'] = spectrum
            results.output['status'] = 1
        except Exception as e:
            results.output['error'] = 'getSpectra: Failed to get spectrum: {0}'.format(str(e))
            results.output['stuff'] = (name, x, y, ra, dec, ext)

        out = json.dumps(results.output)
        results.reset()
        return out

CubeView.register(api)
