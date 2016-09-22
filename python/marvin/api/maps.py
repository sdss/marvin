#!/usr/bin/env python
# encoding: utf-8
#
# maps.py
#
# Created by José Sánchez-Gallego on 25 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import flask.ext.classy
from flask import request

import json

import brain.utils.general
import marvin.api.base
from marvin.api import parse_params
import marvin.core.exceptions
import marvin.tools.maps
import marvin.utils.general


def _getMaps(name, **kwargs):
    """Returns a Maps object after parsing the name."""

    results = {}

    # Makes sure we don't use the wrong mode.
    kwargs.pop('mode', None)

    drpver, dapver = parse_params(request)

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
                                      mode='local', drpver=drpver, dapver=dapver, **kwargs)
        results['status'] = 1
    except Exception as ee:
        maps = None
        results['error'] = 'Failed to retrieve maps {0}: {1}'.format(name, str(ee))

    return maps, results


class MapsView(marvin.api.base.BaseView):
    """Class describing API calls related to MaNGA Maps."""

    route_base = '/maps/'

    def index(self):
        self.results['data'] = 'this is a maps!'
        return json.dumps(self.results)

    @flask.ext.classy.route('/<name>/<bintype>/<template_kin>/',
                            methods=['GET', 'POST'], endpoint='getMaps')
    def get(self, name, bintype, template_kin):
        """Returns the parameters needed to initialise a Maps remotely.

        To initialise a Maps we need to return:
        - mangaid
        - plateifu
        - Header with WCS information
        - Maps shape
        - bintype
        - template_kin

        """

        kwargs = {'bintype': bintype, 'template_kin': template_kin}
        maps, results = _getMaps(name, **kwargs)
        self.update_results(results)

        if maps is None:
            return json.dumps(self.results)

        wcs_header = maps.data.cube.wcs.makeHeader().tostring()
        shape = maps.shape
        bintype = maps.bintype
        template_kin = maps.template_kin

        # Redefines plateifu and mangaid from the Maps
        mangaid = maps.mangaid
        plateifu = maps.plateifu

        self.results['data'] = {name: {'mangaid': mangaid,
                                       'plateifu': plateifu,
                                       'wcs': wcs_header,
                                       'shape': shape,
                                       'bintype': bintype,
                                       'template_kin': template_kin}}

        return json.dumps(self.results)

    @flask.ext.classy.route('/<name>/dap_props/<path:path>',
                            methods=['GET', 'POST'], endpoint='getdap_props')
    @brain.utils.general.parseRoutePath
    def getDAP_props(self, **kwargs):
        """Returns a dictionary of DAP parameters for a Maps spaxel.

        Parameters:
            name (str):
                The ``plateifu`` or ``mangaid`` of the object.
            x,y (int):
                The x/y coordinates of the spaxel (origin is ``lower``).
            kwargs (dict):
                Any other parameter to pass for the ``Maps`` initialisation.

        """

        name = kwargs.pop('name')
        xx = int(kwargs.pop('x'))
        yy = int(kwargs.pop('y'))

        # Initialises the Maps object
        maps, results = _getMaps(name, **kwargs)
        self.update_results(results)

        if maps is None:
            return json.dumps(self.results)

        dict_of_props = marvin.utils.general.dap.maps_db2dict_of_props(
            maps.data, xx, yy)

        self.results['data'] = dict_of_props

        return json.dumps(self.results)

    @flask.ext.classy.route('/<name>/map/<path:path>',
                            methods=['GET', 'POST'], endpoint='getmap')
    @brain.utils.general.parseRoutePath
    def getMap(self, **kwargs):
        """Returns data, ivar, mask, and unit for a given map.

        Parameters:
            name (str):
                The ``plateifu`` or ``mangaid`` of the object.
            property_name (str):
                The property_name of the map to be extractred. E.g., `'emline_gflux'`.
            channel (str or None):
                If the ``property_name`` contains multiple channels, the channel to use,
                e.g., ``ha_6564'. Otherwise, ``None``.

        e.g., https://api.sdss.org/marvin2/api/maps/8485-1901/map/category=emline_gflux/channel=ha_6564/

        """

        name = kwargs.pop('name')
        property_name = str(kwargs.pop('property_name'))
        channel = kwargs.pop('channel', None)

        # Initialises the Maps object
        maps, results = _getMaps(name, **kwargs)
        self.update_results(results)

        if maps is None:
            return json.dumps(self.results)

        try:
            mmap = maps.getMap(property_name=property_name, channel=channel)
            self.results['data'] = {}
            self.results['data']['value'] = mmap.value.tolist()
            self.results['data']['ivar'] = mmap.ivar.tolist()
            self.results['data']['mask'] = mmap.ivar.tolist()
            self.results['data']['unit'] = mmap.unit
            self.results['data']['header'] = {key: mmap.header[key] for key in mmap.header}
        except Exception as ee:
            self.results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(ee))

        return json.dumps(self.results)
