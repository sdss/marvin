#!/usr/bin/env python
# encoding: utf-8
"""

general.py

Licensed under a 3-clause BSD license.

Revision history:
    17 Feb 2016 J. Sánchez-Gallego
      Initial version
    18 Feb 2016 B. Cherinka
        Added buildRouteMap API call

"""

from __future__ import division
from __future__ import print_function

from flask import jsonify
from flask_classy import route

from brain.api.general import BrainGeneralRequestsView
from marvin.utils.general import mangaid2plateifu as mangaid2plateifu
from marvin.utils.general import get_nsa_data
from marvin.api.base import arg_validate as av
import json


class GeneralRequestsView(BrainGeneralRequestsView):

    @route('/mangaid2plateifu/<mangaid>/', endpoint='mangaid2plateifu', methods=['GET', 'POST'])
    @av.check_args()
    def mangaid2plateifu(self, args, mangaid):
        """ Returns a plateifu given a mangaid

        .. :quickref: General; Returns a plateifu given a mangaid

        :param mangaid: The name of the cube as mangaid
        :form release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json string plateifu: id of cube as plateifu
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/general/mangaid2plateifu/1-209232/ HTTP/1.1
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
              "data": {8485-1901"}
           }

        """
        try:
            plateifu = mangaid2plateifu(mangaid, mode='db')
            self.results['data'] = plateifu
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = ('manga2plateifu failed with error: {0}'.format(str(ee)))

        return jsonify(self.results)

    @route('/nsa/full/<mangaid>/', endpoint='nsa_full', methods=['GET', 'POST'])
    @av.check_args()
    def get_nsa_data(self, args, mangaid):
        """Returns the NSA data for a given mangaid from the full catalogue.

        .. :quickref: General; Returns for a given mangaid the NSA data from the full catalog

        :param mangaid: The name of the cube as mangaid
        :form release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json dict nsa_data: dict of all the NSA parameters
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/general/nsa/full/1-209232/ HTTP/1.1
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
              "data": {"aid": 0,
                       "asymmetry":[-0.0217, 0.00517, -0.0187, ...],
                       "bastokes": [[0.7290, 0.9677, ...], [], ...],
                       ...
                      }
           }

        """

        try:
            nsa_data = get_nsa_data(mangaid, mode='local', source='nsa')
            self.results['data'] = nsa_data
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = 'get_nsa_data failed with error: {0}'.format(str(ee))

        # these should be jsonify but switching back to json.dumps until fucking Utah gets with the fucking picture
        return json.dumps(self.results)

    @route('/nsa/drpall/<mangaid>/', endpoint='nsa_drpall', methods=['GET', 'POST'])
    @av.check_args()
    def get_nsa_drpall_data(self, args, mangaid):
        """Returns the NSA data in drpall for a given mangaid.

        .. :quickref: General; Returns for a given mangaid the NSA data from the drpall file

        Note that this always uses the drpver/drpall versions that are default in the server.

        :param mangaid: The name of the cube as mangaid
        :form release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json dict nsa_data: dict of the NSA parameters from the DRPall file
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/general/nsa/drpall/1-209232/ HTTP/1.1
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
              "data": {"camcol": 2,
                       "elpetro_absmag": [-14.3422, -15.7994,-17.0133, ...],
                       "elpetro_amivar": [],
                       ...
                      }
           }

        """

        try:
            nsa_data = get_nsa_data(mangaid, mode='local', source='drpall')
            self.results['data'] = nsa_data
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = 'get_nsa_data failed with error: {0}'.format(str(ee))

        return json.dumps(self.results)
