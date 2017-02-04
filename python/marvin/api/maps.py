#!/usr/bin/env python
# encoding: utf-8
#
# maps.py
#
# Created by José Sánchez-Gallego on 25 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import flask_classy
from flask import request, jsonify, abort

import json

import marvin.api.base
from marvin.api import parse_params
import marvin.core.exceptions
import marvin.tools.maps
import marvin.utils.general
from webargs import fields, validate
from webargs.flaskparser import use_args


def _getMaps(name, **kwargs):
    """Returns a Maps object after parsing the name."""

    results = {}

    # Makes sure we don't use the wrong mode.
    kwargs.pop('mode', None)

    release = parse_params(request)

    # Parses name into either mangaid or plateifu
    try:
        idtype = marvin.utils.general.parseIdentifier(name)
    except Exception as ee:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(ee))
        return None, results

    plateifu = None
    mangaid = None
    try:
        if idtype == 'plateifu':
            plateifu = name
        elif idtype == 'mangaid':
            mangaid = name
        else:
            raise marvin.core.exceptions.MarvinError(
                'invalid plateifu or mangaid: {0}'.format(idtype))

        maps = marvin.tools.maps.Maps(mangaid=mangaid, plateifu=plateifu,
                                      mode='local', release=release, **kwargs)
        results['status'] = 1
    except Exception as ee:
        maps = None
        results['error'] = 'Failed to retrieve maps {0}: {1}'.format(name, str(ee))

    return maps, results

maps_args = {
    'name': fields.String(required=True, location='view_args', validate=validate.Length(min=4)),
    'release': fields.String(required=True, validate=validate.Regexp('MPL-[4-9]')),
    'bintype': fields.String(required=True, location='view_args'),
    'template_kin': fields.String(required=True, location='view_args')
}


class MapsView(marvin.api.base.BaseView):
    """Class describing API calls related to MaNGA Maps."""

    route_base = '/maps/'

    def index(self):
        '''Returns general maps info

        .. :quickref: Maps; Get general maps info

        :form inconfig: json of any incoming parameters
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson string data: data message
        :resheader Content-Type: application/json
        :statuscode 200: no error

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/maps/ HTTP/1.1
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
              "data": "this is a maps!"
           }

        '''
        abort(500)
        self.results['status'] = 1
        self.results['data'] = 'this is a maps!'
        return jsonify(self.results)

    @flask_classy.route('/<name>/<bintype>/<template_kin>/',
                        methods=['GET', 'POST'], endpoint='getMaps')
    @use_args(maps_args)
    def get(self, args, name, bintype, template_kin):
        '''Returns the parameters needed to initialise a Maps remotely.

        .. :quickref: Maps; Get a maps given a name, bintype, and template_kin

        :param name: The name of the maps as plate-ifu or mangaid
        :param bintype: The bintype associated with this maps.  If not defined, the default is used
        :param template_kin: The template_kin associated with this maps.  If not defined, the default is used
        :form inconfig: json of any incoming parameters
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json string plateifu: id of maps
        :json string mangaid: mangaid of maps
        :json string header: the maps header as a string
        :json list shape: the maps shape [x, y]
        :json string wcs: the maps wcs_header as a string
        :json string bintype: the bintype of the maps
        :json string template_kin: the template library of the maps
        :resheader Content-Type: application/json
        :statuscode 200: no error

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/maps/8485-1901/SPX/GAU-MILESHC/ HTTP/1.1
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
              "data": {"8485-1901": {"plateifu": "8485-1901",
                            "mangaid": "1-209232",
                            "header": "XTENSION= 'IMAGE', NAXIS=3, .... END",
                            "wcs_header": "WCSAXES = 3 / Number of coordindate axes .... END",
                            "shape": [34, 34],
                            "bintype": "SPX",
                            "template_kin": "GAU-MILESHC"
                            }
                      }
           }

        '''
        kwargs = {'bintype': bintype, 'template_kin': template_kin}
        maps, results = _getMaps(name, **kwargs)
        self.update_results(results)

        if maps is None:
            return json.dumps(self.results)

        header = maps.header.tostring()
        wcs_header = maps.wcs.to_header_string()
        shape = maps.shape
        bintype = maps.bintype
        template_kin = maps.template_kin

        # Redefines plateifu and mangaid from the Maps
        mangaid = maps.mangaid
        plateifu = maps.plateifu

        self.results['data'] = {name: {'mangaid': mangaid,
                                       'plateifu': plateifu,
                                       'header': header,
                                       'wcs': wcs_header,
                                       'shape': shape,
                                       'bintype': bintype,
                                       'template_kin': template_kin}}

        return jsonify(self.results)

    @flask_classy.route('/<name>/<bintype>/<template_kin>/map/<property_name>/<channel>/',
                        methods=['GET', 'POST'], endpoint='getmap')
    @use_args(maps_args)
    def getMap(self, name, bintype, template_kin, property_name, channel):
        """Returns data, ivar, mask, and unit for a given map.

        Parameters:
            name (str):
                The ``plateifu`` or ``mangaid`` of the object.
            bintype (str):
                The bintype associated with this model cube. If not defined,
                the default type of binning will be used.
            template_kin (str):
                The template_kin associated with this model cube.
                If not defined, the default template_kin will be used.
            property_name (str):
                The property_name of the map to be extractred. E.g., `'emline_gflux'`.
            channel (str or None):
                If the ``property_name`` contains multiple channels, the channel to use,
                e.g., ``ha_6564'. Otherwise, ``None``.

        e.g., https://api.sdss.org/marvin2/api/maps/8485-1901/SPX/GAU-MILESHC/map/emline_gflux/ha_6564/

        """

        kwargs = {'bintype': bintype, 'template_kin': template_kin}

        # Initialises the Maps object
        maps, results = _getMaps(name, **kwargs)
        self.update_results(results)

        if maps is None:
            return json.dumps(self.results)

        try:
            mmap = maps.getMap(property_name=str(property_name), channel=str(channel))
            self.results['data'] = {}
            self.results['data']['value'] = mmap.value.tolist()
            self.results['data']['ivar'] = mmap.ivar.tolist() if mmap.ivar is not None else None
            self.results['data']['mask'] = mmap.mask.tolist() if mmap.mask is not None else None
            self.results['data']['unit'] = mmap.unit
            self.results['data']['header'] = {key: mmap.header[key] for key in mmap.header}
        except Exception as ee:
            self.results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(ee))

        return json.dumps(self.results)

    @flask_classy.route('/<name>/<bintype>/<template_kin>/spaxels/<binid>',
                        methods=['GET', 'POST'], endpoint='getbinspaxels')
    def getBinSpaxels(self, name, bintype, template_kin, binid):
        """Returns a list of x and y indices for spaxels belonging to ``binid``.

        Parameters:
            name (str):
                The ``plateifu`` or ``mangaid`` of the object.
            bintype (str):
                The bintype associated with this model cube. If not defined,
                the default type of binning will be used.
            template_kin (str):
                The template_kin associated with this model cube.
                If not defined, the default template_kin will be used.
            binid (int):
                The binid to which the spaxels belong.

        e.g., https://api.sdss.org/marvin2/api/maps/8485-1901/SPX/GAU-MILESHC/spaxels/112/

        """

        kwargs = {'bintype': bintype, 'template_kin': template_kin}

        # Initialises the Maps object
        maps, results = _getMaps(name, **kwargs)
        self.update_results(results)

        if maps is None:
            return json.dumps(self.results)

        try:
            self.results['data'] = {}
            self.results['data']['spaxels'] = maps.get_bin_spaxels(binid, only_list=True)
        except Exception as ee:
            self.results['error'] = ('Failed to get spaxels for binid={0}: {1}'
                                     .format(binid, str(ee)))

        return json.dumps(self.results)
