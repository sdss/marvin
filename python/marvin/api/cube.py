from flask.ext.classy import route
from flask import Blueprint, url_for, current_app, request
from marvin.tools.cube import Cube
from marvin.api.base import BaseView
import json
import urllib
from marvin.utils.general import parseRoutePath, parseName

''' stuff that runs server-side '''

api = Blueprint("api", __name__)


def _getCube(name):
    ''' Retrieve a cube using marvin tools '''

    cube = None
    results = {}

    # parse name into either mangaid or plateifu
    try:
        mangaid, plateifu = parseName(name)
    except Exception as e:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(e))
        return cube, results

    try:
        cube = Cube(mangaid=mangaid, plateifu=plateifu, mode='local')
        results['status'] = 1
    except Exception as e:
        results['error'] = 'Failed to retrieve cube {0}: {1}'.format(name, str(e))

    return cube, results


class CubeView(BaseView):
    ''' Class describing API calls related to MaNGA Cubes '''

    route_base = '/cubes/'
    # decorators = [parseRoutePath]

    def index(self):
        self.results['data'] = 'this is a cube!'
        return json.dumps(self.results)
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
        ''' This method performs a get request at the url route /cubes/<id> '''
        cube, res = _getCube(name)
        self.update_results(res)
        if cube:
            self.results['data'] = {name: '{0},{1},{2},{3}'.format(name, cube.plate, cube.ra, cube.dec)}
        return json.dumps(self.results)

    @route('/<name>/spectra', defaults={'path': None}, methods=['GET', 'POST'])
    @route('/<name>/spectra/<path:path>', endpoint='getspectra', methods=['GET', 'POST'])
    @parseRoutePath
    def getSpectra(self, name=None, x=None, y=None, ra=None, dec=None, ext=None):
        # Add ability to grab spectra from fits files
        cube, res = _getCube(name)
        self.update_results(res)
        if not cube:
            self.results['error'] = 'getSpectra: No cube: {0}'.format(res['error'])
            return json.dumps(self.results)

        try:
            spectrum = cube.getSpectrum(x=x, y=y, ra=ra, dec=dec, ext=ext)
            self.results['data'] = spectrum
            self.results['status'] = 1
        except Exception as e:
            self.results['status'] = -1
            self.results['error'] = 'getSpectra: Failed to get spectrum: {0}'.format(str(e))
            self.results['stuff'] = (name, x, y, ra, dec, ext)

        return json.dumps(self.results)

CubeView.register(api)
