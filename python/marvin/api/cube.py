#!/usr/bin/env python
# encoding: utf-8

import os

import numpy as np

from flask_classful import route
from flask import jsonify, request, Response
import json

from marvin import config
from marvin.api.base import BaseView, arg_validate as av
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier, mangaid2plateifu
from marvin.tools.cube import Cube

from brain.core.exceptions import BrainError

try:
    from sdss_access.path import Path
except ImportError:
    Path = None


''' stuff that runs server-side '''


def _getCube(name, use_file=False, release=None, **kwargs):
    ''' Retrieve a cube using marvin tools '''

    drpver, __ = config.lookUpVersions(release)

    cube = None
    results = {}

    # parse name into either mangaid or plateifu
    try:
        idtype = parseIdentifier(name)
    except Exception as ee:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(ee))
        return cube, results

    filename = None
    plateifu = None
    mangaid = None

    try:
        if use_file:

            if idtype == 'mangaid':
                plate, ifu = mangaid2plateifu(name, drpver=drpver)
            elif idtype == 'plateifu':
                plate, ifu = name.split('-')

            if Path is not None:
                filename = Path().full('mangacube', ifu=ifu, plate=plate, drpver=drpver)
                assert os.path.exists(filename), 'file not found.'
            else:
                raise MarvinError('cannot create path for MaNGA cube.')

        else:

            if idtype == 'plateifu':
                plateifu = name
            elif idtype == 'mangaid':
                mangaid = name
            else:
                raise MarvinError('invalid plateifu or mangaid: {0}'.format(idtype))

        cube = Cube(filename=filename, mangaid=mangaid, plateifu=plateifu,
                    mode='local', release=release)

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

           GET /marvin/api/cubes/ HTTP/1.1
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
        av.manual_parse(self, request)
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
        :json list wavelength: the cube wavelength array
        :json string wcs_header: the cube wcs_header as a string
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin/api/cubes/8485-1901/ HTTP/1.1
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

            wavelength = (cube._wavelength.tolist() if isinstance(cube._wavelength, np.ndarray)
                          else cube._wavelength)
            self.results['data'] = {'plateifu': name,
                                    'mangaid': cube.mangaid,
                                    'ra': cube.ra,
                                    'dec': cube.dec,
                                    'header': cube.header.tostring(),
                                    'redshift': nsa_data.z if nsa_data else -9999,
                                    'wavelength': wavelength,
                                    'wcs_header': cube.wcs.to_header_string(),
                                    'shape': cube._shape}

        return jsonify(self.results)

    @route('/<name>/extensions/<cube_extension>/', methods=['GET', 'POST'],
           endpoint='getExtension')
    @av.check_args()
    def getExtension(self, args, name, cube_extension):
        """Returns the extension for a cube given a plateifu/mangaid.

        .. :quickref: Cube; Gets a specified cube extension for a given plate-ifu or mangaid

        :param name: The name of the cube as plate-ifu or mangaid
        :param cube_extension: The name of the cube extension.  Either flux, ivar, or mask.
        :form release: the release of MaNGA
        :form use_file: if True, forces to load the cube from a file.
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json string cube_extension: the data for the specified extension
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin/api/cubes/8485-1901/extensions/flux/ HTTP/1.1
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
              "data": {"extension_data": [[0,0,..0], [], ... [0, 0, 0,... 0]]
              }
           }
        """

        # Pass the args in and get the cube
        args = self._pop_args(args, arglist=['name', 'cube_extension'])
        cube, res = _getCube(name, use_file=True, **args)
        self.update_results(res)

        if cube:

            extension_data = cube.data[cube_extension.upper()].data

            if extension_data is None:
                self.results['data'] = {'extension_data': None}
            else:
                self.results['data'] = {'extension_data': extension_data.tolist()}

        return Response(json.dumps(self.results), mimetype='application/json')

    @route('/<name>/quantities/<x>/<y>/', methods=['GET', 'POST'],
           endpoint='getCubeQuantitiesSpaxel')
    @av.check_args()
    def getCubeQuantitiesSpaxel(self, args, name, x, y):
        """Returns a dictionary with all the quantities.

        .. :quickref: Cube; Returns a dictionary with all the quantities

        :param name: The name of the cube as plate-ifu or mangaid
        :param x: The x coordinate of the spaxel (origin is ``lower``)
        :param y: The y coordinate of the spaxel (origin is ``lower``)
        :form release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin/api/cubes/8485-1901/quantities/10/12/ HTTP/1.1
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
              "data": {"flux": {"value": [0,0,..0], "ivar": ...},
                       "specres": ...}
              }
           }
        """

        # Pass the args in and get the cube
        args = self._pop_args(args, arglist=['name', 'x', 'y'])
        cube, res = _getCube(name, **args)
        self.update_results(res)

        if cube:

            self.results['data'] = {}

            spaxel_quantities = cube._get_spaxel_quantities(x, y)

            for quant in spaxel_quantities:

                spectrum = spaxel_quantities[quant]

                if spectrum is None:
                    self.data[quant] = {'value': None}
                    continue

                value = spectrum.value.tolist()
                ivar = spectrum.ivar.tolist() if spectrum.ivar is not None else None
                mask = spectrum.mask.tolist() if spectrum.mask is not None else None

                self.results['data'][quant] = {'value': value,
                                               'ivar': ivar,
                                               'mask': mask}

            self.results['data']['wavelength'] = cube._wavelength.tolist()

        return Response(json.dumps(self.results), mimetype='application/json')
