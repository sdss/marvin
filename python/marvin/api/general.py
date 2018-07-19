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

from flask import jsonify, Response, request
from flask_classful import route

from brain.api.base import processRequest
from brain.api.general import BrainGeneralRequestsView
from brain.utils.general import validate_user, get_db_user
from brain.utils.general.decorators import public
from marvin import marvindb, config
from marvin.utils.general import mangaid2plateifu as mangaid2plateifu
from marvin.utils.general import get_nsa_data
from marvin.api.base import arg_validate as av
from flask_jwt_extended import create_access_token
import json


def get_drpver(release=None):
    ''' Get the drpver from the input release from the request '''
    if not release:
        release = config.release

    drpver, dapver = config.lookUpVersions(release)
    return drpver


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

           GET /marvin/api/general/mangaid2plateifu/1-209232/ HTTP/1.1
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
        # get drpver
        drpver = get_drpver(release=args.get('release', None))

        try:
            plateifu = mangaid2plateifu(mangaid, mode='db', drpver=drpver)
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

           GET /marvin/api/general/nsa/full/1-209232/ HTTP/1.1
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
        # get drpver
        drpver = get_drpver(release=args.get('release', None))

        try:
            nsa_data = get_nsa_data(mangaid, mode='local', source='nsa', drpver=drpver)
            self.results['data'] = nsa_data
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = 'get_nsa_data failed with error: {0}'.format(str(ee))

        # these should be jsonify but switching back to json.dumps until fucking Utah gets with the fucking picture
        return Response(json.dumps(self.results), mimetype='application/json')

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

           GET /marvin/api/general/nsa/drpall/1-209232/ HTTP/1.1
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
        # get drpver
        drpver = get_drpver(release=args.get('release', None))

        try:
            nsa_data = get_nsa_data(mangaid, mode='local', source='drpall', drpver=drpver)
            self.results['data'] = nsa_data
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = 'get_nsa_data failed with error: {0}'.format(str(ee))

        return Response(json.dumps(self.results), mimetype='application/json')

    @public
    @route('/login/', methods=['POST'], endpoint='login')
    def login(self):
        ''' Server-Side login to generate a new token '''

        result = {}

        # check the form
        form = processRequest(request=request)
        if form is None:
            result['error'] = 'Request has no form data!'
            return jsonify(result), 400

        # get username and password
        username = form.get('username', None)
        password = form.get('password', None)
        # return if no valid login form data
        if not username or not password:
            result['error'] = 'Missing username and/or password!'
            return jsonify(result), 400
        username = username.strip()
        password = password.strip()

        # validate the user with htpassfile or trac username
        is_valid, user, result = validate_user(username, password, request=request)

        # User code goes here
        if is_valid:
            user = get_db_user(username, password, dbsession=marvindb.session, user_model=marvindb.datadb.User, request=request)
            if user and user.check_password(password):
                # generate token if valid
                access_token = create_access_token(identity=user.username, fresh=True)
                return jsonify(access_token=access_token), 200
        else:
            msg = result['error'] if 'error' in result else ''
            result['error'] = 'Not valid login. Bad username or password. {0}'.format(msg)
            return jsonify(result), 401
