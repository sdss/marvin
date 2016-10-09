import json

from flask_classy import route
from flask import Blueprint, redirect, url_for
from flask import request

from marvin.api import parse_params
from marvin.api.base import BaseView
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier
from marvin.tools.cube import Cube

from brain.utils.general import parseRoutePath

''' stuff that runs server-side '''

# api = Blueprint("api", __name__)


def _getCube(name):
    ''' Retrieve a cube using marvin tools '''

    # Gets the drpver from the request
    release = parse_params(request)

    cube = None
    results = {}

    # parse name into either mangaid or plateifu
    try:
        idtype = parseIdentifier(name)
    except Exception as ee:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(ee))
        return cube, results

    try:
        if idtype == 'plateifu':
            plateifu = name
            mangaid = None
        elif idtype == 'mangaid':
            mangaid = name
            plateifu = None
        else:
            raise MarvinError('invalid plateifu or mangaid: {0}'.format(idtype))

        cube = Cube(mangaid=mangaid, plateifu=plateifu, mode='local', release=release)
        results['status'] = 1
    except Exception as ee:
        results['error'] = 'Failed to retrieve cube {0}: {1}'.format(name, str(ee))

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
        """Returns the necessary information to instantiate a cube for a given plateifu."""

        cube, res = _getCube(name)
        self.update_results(res)
        if cube:
            self.results['data'] = {name: '{0},{1},{2},{3}'.format(name, cube.plate,
                                                                   cube.ra, cube.dec),
                                    'header': cube.header.tostring(),
                                    'redshift': cube.data.target.NSA_objects[0].z,
                                    'shape': cube.shape,
                                    'wavelength': cube.wavelength,
                                    'wcs_header': cube.data.wcs.makeHeader().tostring()}

        return json.dumps(self.results)

    # TODO: This is not used anymore, so maybe it should be removed.
    @route('/<name>/spectra/', methods=['GET', 'POST'], endpoint='allspectra')
    def getAllSpectra(self, name=None):
        ''' placeholder to retrieve all spectra for a given cube.  For now, do nothing '''
        self.results['data'] = '{0}, {1}'.format(name, url_for('api.getspectra', name=name, path=''))
        return json.dumps(self.results)

    # TODO: This is not used anymore, so maybe it should be removed.
    @route('/<name>/spaxels/<path:path>', methods=['GET', 'POST'], endpoint='getspaxels')
    @parseRoutePath
    def getSpaxels(self, **kwargs):
        """Returns a list of x, y positions for all the spaxels in a given cube."""

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
