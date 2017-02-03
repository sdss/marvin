import json

import numpy as np

from flask_classy import route
from flask import Blueprint, redirect, url_for
from flask import request, jsonify

from marvin.api import parse_params
from marvin.api.base import BaseView
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier
from marvin.tools.cube import Cube

from brain.utils.general import parseRoutePath
from brain.core.exceptions import BrainError

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
        '''Returns general cube info

        .. :quickref: Cube; Get general cube info

        :query string release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson string data: data message
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/cubes/ HTTP/1.1
           Host: api.sdss.org
           Accept: application/json, */*

        **Example response**:

        .. sourcecode:: http

           HTTP/1.1 200 OK
           Content-Type: application/json
           {
              "status": 1,
              "error": null,
              "inconfig": {"release": "MPL-5"},
              "utahconfig": {"release": "MPL-5", "mode": "local"},
              "traceback": null,
              "data": "this is a cube!"
           }

        '''
        self.results['status'] = 1
        self.results['data'] = 'this is a cube!'
        return jsonify(self.results)

    @route('/<name>/', methods=['GET', 'POST'], endpoint='getCube')
    def get(self, name):
        '''Returns the necessary information to instantiate a cube for a given plateifu.

        .. :quickref: Cube; Get a cube given a plate-ifu or mangaid

        :param name: The name of the cube as plate-ifu or mangaid
        :form release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json string plateifu: id of cube
        :json string mangaid: mangaid of cube
        :json float ra: RA of cube
        :json float dec: Dec of cube
        :json string header: the cube header as a string
        :json float redshift: the cube redshift
        :json list shape: the cube shape [x, y]
        :json list wavelength: the cube wavelength array
        :json string wcs_header: the cube wcs_header as a string
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/cubes/8485-1901/ HTTP/1.1
           Host: api.sdss.org
           Accept: application/json, */*

        **Example response**:

        .. sourcecode:: http

           HTTP/1.1 200 OK
           Content-Type: application/json
           {
              "status": 1,
              "error": null,
              "inconfig": {"release": "MPL-5"},
              "utahconfig": {"release": "MPL-5", "mode": "local"},
              "traceback": null,
              "data": {"plateifu": "8485-1901",
                    "mangaid": "1-209232",
                    "ra": 232.544703894,
                    "dec": 48.6902009334,
                    "header": "XTENSION= 'IMAGE', NAXIS=3, .... END",
                    "wcs_header": "WCSAXES = 3 / Number of coordindate axes .... END",
                    "redshift": 0.0407447,
                    "shape": [34, 34],
                    "wavelength": [3621.6, 3622.43,...,10353.8]
              }
           }

        '''

        cube, res = _getCube(name)
        self.update_results(res)

        if cube:

            try:
                nsa_data = cube.nsa
            except (MarvinError, BrainError):
                nsa_data = None

            wavelength = (cube.wavelength.tolist() if isinstance(cube.wavelength, np.ndarray)
                          else cube.wavelength)
            self.results['data'] = {name: '{0}, {1}, {2}, {3}'.format(name, cube.plate,
                                                                      cube.ra, cube.dec),
                                    'header': cube.header.tostring(),
                                    'redshift': nsa_data.z if nsa_data else -9999,
                                    'shape': cube.shape,
                                    'wavelength': wavelength,
                                    'wcs_header': cube.wcs.to_header_string()}

        return jsonify(self.results)

    # TODO: This is not used anymore, so maybe it should be removed.
    @route('/<name>/spectra/', methods=['GET', 'POST'], endpoint='allspectra')
    def getAllSpectra(self, name=None):
        ''' placeholder to retrieve all spectra for a given cube.  For now, do nothing

        .. :quickref: Cube; Get a spectrum from a cube

        '''
        self.results['data'] = '{0}, {1}'.format(name, url_for('api.getspectra', name=name, path=''))
        return json.dumps(self.results)

    # TODO: This is not used anymore, so maybe it should be removed.
    @route('/<name>/spaxels/<path:path>', methods=['GET', 'POST'], endpoint='getspaxels')
    @parseRoutePath
    def getSpaxels(self, **kwargs):
        """Returns a list of x, y positions for all the spaxels in a given cube.

        .. :quickref: Cube; Get a spaxel from a cube

        """

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
