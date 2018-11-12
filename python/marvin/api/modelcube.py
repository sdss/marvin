#!/usr/bin/env python
# encoding: utf-8
#
# modelcube.py
#
# Licensed under a 3-clause BSD license.
#
# Revision history:
#     25 Sep 2016 J. Sánchez-Gallego
#       Initial version


from __future__ import division, print_function

import json
import os

from flask import Response, jsonify
from flask_classful import route

from marvin import config
from marvin.api.base import BaseView
from marvin.api.base import arg_validate as av
from marvin.core.exceptions import MarvinError
from marvin.tools.modelcube import ModelCube
from marvin.utils.general import mangaid2plateifu, parseIdentifier


try:
    from sdss_access.path import Path
except ImportError:
    Path = None


def _get_model_cube(name, use_file=False, release=None, **kwargs):
    """Retrieves a Marvin ModelCube object."""

    model_cube = None
    results = {}

    drpver, dapver = config.lookUpVersions(release)

    # parse name into either mangaid or plateifu
    try:
        idtype = parseIdentifier(name)
    except Exception as err:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(err))
        return model_cube, results

    filename = None
    plateifu = None
    mangaid = None

    bintype = kwargs.pop('bintype')
    template = kwargs.pop('template')

    try:
        if use_file:

            if idtype == 'mangaid':
                plate, ifu = mangaid2plateifu(name, drpver=drpver)
            elif idtype == 'plateifu':
                plate, ifu = name.split('-')

            if Path is not None:

                daptype = '{0}-{1}'.format(bintype, template)

                filename = Path().full('mangadap5', ifu=ifu,
                                       drpver=drpver,
                                       dapver=dapver,
                                       plate=plate, mode='LOGCUBE',
                                       daptype=daptype)
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

        model_cube = ModelCube(filename=filename, mangaid=mangaid, plateifu=plateifu,
                               release=release, template=template, bintype=bintype, **kwargs)

        results['status'] = 1

    except Exception as err:

        results['error'] = 'Failed to retrieve ModelCube {0}: {1}'.format(name, str(err))

    return model_cube, results


class ModelCubeView(BaseView):
    """Class describing API calls related to ModelCubes."""

    route_base = '/modelcubes/'

    @route('/<name>/', defaults={'bintype': None, 'template': None},
           methods=['GET', 'POST'], endpoint='getModelCube')
    @route('/<name>/<bintype>/', defaults={'template': None},
           methods=['GET', 'POST'], endpoint='getModelCube')
    @route('/<name>/<bintype>/<template>/', methods=['GET', 'POST'], endpoint='getModelCube')
    @av.check_args()
    def get(self, args, name, bintype, template):
        """Retrieves a ModelCube.

        .. :quickref: ModelCube; Get a modelcube given a name, bintype, and template

        :param name: The name of the modelcube as plate-ifu or mangaid
        :param bintype: The bintype associated with this modelcube.
                        If not defined, the default is used
        :param template: The template associated with this modelcube.
                         If not defined, the default is used
        :form release: the release of MaNGA data
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json string header: the modelcube header as a string
        :json list shape: the modelcube shape [x, y]
        :json list wavelength: the modelcube wavelength array
        :json list redcorr: the modelcube redcorr array
        :json string wcs_header: the modelcube wcs_header as a string
        :json string bintype: the bintype of the modelcube
        :json string template: the template library of the modelcube
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin/api/modelcubes/8485-1901/SPX/GAU-MILESHC/ HTTP/1.1
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
              "data": {"plateifu": 8485-1901,
                       "mangaid": '1-209232',
                       "header": "XTENSION= 'IMAGE', NAXIS=3, .... END",
                       "wcs_header": "WCSAXES = 3 / Number of coordindate axes .... END",
                       "shape": [34, 34],
                       "wavelength": [3621.6, 3622.43, 3623.26, ...],
                       "redcorr": [1.06588, 1.065866, 1.06585, ...],
                       "bintype": "SPX",
                       "template": "GAU-MILESHC"
                      }
           }

        """

        # Pop any args we don't want going into ModelCube
        args = self._pop_args(args, arglist='name')

        model_cube, results = _get_model_cube(name, **args)

        self.update_results(results)

        if model_cube is not None:
            self.results['data'] = {
                'plateifu': model_cube.plateifu,
                'mangaid': model_cube.mangaid,
                'header': model_cube.header.tostring(),
                'shape': model_cube._shape,
                'wavelength': model_cube._wavelength.tolist(),
                'redcorr': model_cube._redcorr.tolist(),
                'wcs_header': model_cube.wcs.to_header_string(),
                'bintype': model_cube.bintype.name,
                'template': model_cube.template.name}

        return jsonify(self.results)

    @route('/<name>/extensions/<modelcube_extension>/',
           defaults={'bintype': None, 'template': None},
           methods=['GET', 'POST'], endpoint='getModelCubeExtension')
    @route('/<name>/<bintype>/extensions/<modelcube_extension>/',
           defaults={'template': None},
           methods=['GET', 'POST'], endpoint='getModelCubeExtension')
    @route('/<name>/<bintype>/<template>/extensions/<modelcube_extension>/',
           methods=['GET', 'POST'], endpoint='getModelCubeExtension')
    @av.check_args()
    def getModelCubeExtension(self, args, name, bintype, template, modelcube_extension):
        """Returns the extension for a modelcube given a plateifu/mangaid.

        .. :quickref: ModelCube; Gets the extension given a plate-ifu or mangaid

        :param name: The name of the cube as plate-ifu or mangaid
        :param bintype: The bintype associated with this modelcube.
        :param template: The template associated with this modelcube.
        :param modelcube_extension: The name of the cube extension.  Either flux, ivar, or mask.
        :form release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json string modelcube_extension: the data for the specified extension
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin/api/modelcubes/8485-1901/extensions/flux/ HTTP/1.1
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
        args = self._pop_args(args, arglist=['name', 'modelcube_extension'])
        modelcube, res = _get_model_cube(name, use_file=True, **args)
        self.update_results(res)

        if modelcube:

            extension_data = modelcube.data[modelcube_extension.upper()].data

            if extension_data is None:
                self.results['data'] = {'extension_data': None}
            else:
                self.results['data'] = {'extension_data': extension_data.tolist()}

        return Response(json.dumps(self.results), mimetype='application/json')

    @route('/<name>/binids/<modelcube_extension>/',
           defaults={'bintype': None, 'template': None},
           methods=['GET', 'POST'], endpoint='getModelCubeBinid')
    @route('/<name>/<bintype>/binids/<modelcube_extension>/',
           defaults={'template': None},
           methods=['GET', 'POST'], endpoint='getModelCubeBinid')
    @route('/<name>/<bintype>/<template>/binids/<modelcube_extension>/',
           methods=['GET', 'POST'], endpoint='getModelCubeBinid')
    @av.check_args()
    def getModelCubeBinid(self, args, name, bintype, template, modelcube_extension):
        """Returns the binid array for a modelcube given a plateifu/mangaid.

        .. :quickref: ModelCube; Gets the binid array given a plate-ifu or mangaid

        :param name: The name of the modelcube as plate-ifu or mangaid
        :param bintype: The bintype associated with this modelcube.
        :param template: The template associated with this modelcube.
        :param modelcube_extension: The name of the cube extension.
        :form release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json string binid: the binid data
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin/api/modelcubes/8485-1901/binids/flux/ HTTP/1.1
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
              "data": {"binid": [[0,0,..0], [], ... [0, 0, 0,... 0]]
              }
           }
        """

        # Pass the args in and get the cube
        args = self._pop_args(args, arglist=['name', 'modelcube_extension'])
        modelcube, res = _get_model_cube(name, use_file=False, **args)
        self.update_results(res)

        if modelcube:
            try:
                model = modelcube.datamodel.from_fits_extension(modelcube_extension)
                binid_data = modelcube.get_binid(model)
                self.results['data'] = {'binid': binid_data.value.tolist()}
            except Exception as ee:
                self.results['error'] = str(ee)

        return Response(json.dumps(self.results), mimetype='application/json')

    @route('/<name>/<bintype>/<template>/quantities/<x>/<y>/',
           methods=['GET', 'POST'], endpoint='getModelCubeQuantitiesSpaxel')
    @av.check_args()
    def getModelCubeQuantitiesSpaxel(self, args, name, bintype, template, x, y):
        """Returns a dictionary with all the quantities.

        .. :quickref: ModelCube; Returns a dictionary with all the quantities

        :param name: The name of the cube as plate-ifu or mangaid
        :param bintype: The bintype associated with this modelcube.
        :param template: The template associated with this modelcube.
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

           GET /marvin/api/modelcubes/8485-1901/SPX/GAU_MILESHC/quantities/10/12/ HTTP/1.1
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
              "data": {"binned_flux": {"value": [0,0,..0], "ivar": ...},
                       "emline_fit": ...}
              }
           }
        """

        # Pass the args in and get the cube
        args = self._pop_args(args, arglist=['name', 'x', 'y'])
        modelcube, res = _get_model_cube(name, **args)
        self.update_results(res)

        if modelcube:

            self.results['data'] = {}

            spaxel_quantities = modelcube._get_spaxel_quantities(x, y)

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

            self.results['data']['wavelength'] = modelcube._wavelength.tolist()

        return Response(json.dumps(self.results), mimetype='application/json')
