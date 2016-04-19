from flask.ext.classy import route
from flask import Blueprint, redirect, url_for
from marvin.tools.cube import Cube
from marvin.api.base import BaseView
import json
from marvin.utils.general import parseRoutePath, parseName

''' stuff that runs server-side '''

# api = Blueprint("api", __name__)


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

    @route('/<name>/', methods=['GET', 'POST'], endpoint='getCube')
    def get(self, name):
        ''' This method performs a get request at the url route /cubes/<id> '''
        cube, res = _getCube(name)
        self.update_results(res)
        if cube:
            self.results['data'] = {name: '{0},{1},{2},{3}'.format(name, cube.plate, cube.ra, cube.dec)}
        return json.dumps(self.results)

    @route('/<name>/spectra/', methods=['GET', 'POST'], endpoint='allspectra')
    def getAllSpectra(self, name=None):
        ''' placeholder to retrieve all spectra for a given cube.  For now, do nothing '''
        self.results['data'] = '{0}, {1}'.format(name, url_for('api.getspectra', name=name, path=''))
        return json.dumps(self.results)

    @route('/<name>/spaxels/<path:path>', methods=['GET', 'POST'],
           endpoint='getspaxels')
    @parseRoutePath
    def getSpaxels(self, **kwargs):

        name = kwargs.pop('name')
        for var in ['x', 'y', 'ra', 'dec']:
            if var in kwargs:
                kwargs[var] = eval(kwargs[var])

        # Add ability to grab spectra from fits files
        cube, res = _getCube(name)
        self.update_results(res)
        if not cube:
            self.results['error'] = 'getSpaxels: No cube: {0}'.format(
                res['error'])
            return json.dumps(self.results)

        try:
            spaxels = cube.getSpaxel(**kwargs)
            self.results['data'] = {}
            self.results['data']['x'] = [spaxel.x for spaxel in spaxels]
            self.results['data']['y'] = [spaxel.y for spaxel in spaxels]
            self.results['status'] = 1
        except Exception as e:
            self.results['status'] = -1
            self.results['error'] = 'getSpaxels: {0}'.format(str(e))

        return json.dumps(self.results)


    # could not figure out this route, always get BuildError when trying to do a url_for('allspectra'), with the defaults path=''
    # @route('/<name>/spectra/', defaults={'path': ''}, methods=['GET', 'POST'], endpoint='allspectra')
    @route('/<name>/spectra/<path:path>', methods=['GET', 'POST'], endpoint='getspectra')
    @parseRoutePath
    def getSpectra(self, **kwargs):

        name = kwargs.pop('name')

        # Add ability to grab spectra from fits files
        cube, res = _getCube(name)
        self.update_results(res)
        if not cube:
            self.results['error'] = 'getSpectra: No cube: {0}'.format(res['error'])
            return json.dumps(self.results)

        try:
            spectrum = cube.getSpectrum(**kwargs)
            self.results['data'] = spectrum.tolist()
            self.results['status'] = 1
        except Exception as e:
            self.results['status'] = -1
            self.results['error'] = 'getSpectra: Failed to get spectrum: {0}'.format(str(e))

        return json.dumps(self.results)


# CubeView.register(api)
