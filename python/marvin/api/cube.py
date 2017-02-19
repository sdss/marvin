#!/usr/bin/env python
# encoding: utf-8

import numpy as np

from flask_classy import route
from flask import jsonify, request

from marvin.api.base import BaseView, arg_validate as av
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier
from marvin.tools.cube import Cube

from brain.utils.general import parseRoutePath
from brain.core.exceptions import BrainError

''' stuff that runs server-side '''


def _getCube(name, **kwargs):
    ''' Retrieve a cube using marvin tools '''

    # Pop the release to remove a duplicate input to Maps
    release = kwargs.pop('release', None)

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
    # decorators = [av.check_args()]

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
        args = av.manual_parse(self, request)
        self.results['status'] = 1
        self.results['data'] = 'this is a cube!'
        return jsonify(self.results)

    @route('/<name>/', methods=['GET', 'POST'], endpoint='getCube')
    @av.check_args()
    def get(self, args, name):
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

        # Pop any args we don't want going into Cube
        args = self._pop_args(args, arglist='name')

        cube, res = _getCube(name, **args)
        self.update_results(res)

        if cube:

            try:
                nsa_data = cube.nsa
            except (MarvinError, BrainError):
                nsa_data = None

            wavelength = (cube.wavelength.tolist() if isinstance(cube.wavelength, np.ndarray)
                          else cube.wavelength)
            self.results['data'] = {'plateifu': name,
                                    'mangaid': cube.mangaid,
                                    'ra': cube.ra,
                                    'dec': cube.dec,
                                    'header': cube.header.tostring(),
                                    'redshift': nsa_data.z if nsa_data else -9999,
                                    'shape': cube.shape,
                                    'wavelength': wavelength,
                                    'wcs_header': cube.wcs.to_header_string()}

        return jsonify(self.results)

