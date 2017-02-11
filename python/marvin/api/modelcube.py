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


from __future__ import division
from __future__ import print_function

from flask_classy import route
from flask import jsonify

from marvin.tools.modelcube import ModelCube
from marvin.api.base import BaseView, arg_validate as av
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier


def _get_model_cube(name, **kwargs):
    """Retrieves a Marvin ModelCube object."""

    model_cube = None
    results = {}

    # Pop the release to remove a duplicate input to Maps
    release = kwargs.pop('release', None)

    # parse name into either mangaid or plateifu
    try:
        idtype = parseIdentifier(name)
    except Exception as err:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(err))
        return model_cube, results

    try:
        if idtype == 'plateifu':
            plateifu = name
            mangaid = None
        elif idtype == 'mangaid':
            mangaid = name
            plateifu = None
        else:
            raise MarvinError('invalid plateifu or mangaid: {0}'.format(idtype))

        model_cube = ModelCube(mangaid=mangaid, plateifu=plateifu,
                               release=release, **kwargs)
        results['status'] = 1
    except Exception as err:
        results['error'] = 'Failed to retrieve ModelCube {0}: {1}'.format(name, str(err))

    return model_cube, results


class ModelCubeView(BaseView):
    """Class describing API calls related to ModelCubes."""

    route_base = '/modelcubes/'

    @route('/<name>/<bintype>/<template_kin>/', methods=['GET', 'POST'], endpoint='getModelCube')
    @av.check_args()
    def get(self, args, name, bintype, template_kin):
        """Retrieves a ModelCube.

        .. :quickref: ModelCube; Get a modelcube given a name, bintype, and template_kin

        :param name: The name of the maps as plate-ifu or mangaid
        :param bintype: The bintype associated with this maps.  If not defined, the default is used
        :param template_kin: The template_kin associated with this maps.  If not defined, the default is used
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
        :json string template_kin: the template library of the modelcube
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/modelcubes/8485-1901/SPX/GAU-MILESHC/ HTTP/1.1
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
              "data": {"header": "XTENSION= 'IMAGE', NAXIS=3, .... END",
                       "wcs_header": "WCSAXES = 3 / Number of coordindate axes .... END",
                       "shape": [34, 34],
                       "wavelength": [3621.6, 3622.43, 3623.26, ...],
                       "redcorr": [1.06588, 1.065866, 1.06585, ...],
                       "bintype": "SPX",
                       "template_kin": "GAU-MILESHC"
                      }
           }

        """

        # Pop any args we don't want going into ModelCube
        args = self._pop_args(args, arglist='name')

        model_cube, results = _get_model_cube(name, **args)

        self.update_results(results)

        if model_cube is not None:
            self.results['data'] = {
                'header': model_cube.header.tostring(),
                'shape': model_cube.shape,
                'wavelength': model_cube.wavelength.tolist(),
                'redcorr': model_cube.redcorr.tolist(),
                'wcs_header': model_cube.wcs.to_header_string(),
                'bintype': model_cube.bintype,
                'template_kin': model_cube.template_kin}

        return jsonify(self.results)
