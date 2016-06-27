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
import json

import brain.utils.general
import marvin.api.base
import marvin.core.exceptions
import marvin.tools.maps
import marvin.utils.general


def _getMaps(name, **kwargs):
    """Returns a Maps object after parsing the name."""

    results = {}

    # Makes sure we don't use the wrong mode.
    kwargs.pop('mode', None)

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
                                      mode='local', **kwargs)
        results['status'] = 1
    except Exception as ee:
        results['error'] = 'Failed to retrieve maps {0}: {1}'.format(name, str(ee))

    return maps, results


class MapsView(marvin.api.base.BaseView):
    """Class describing API calls related to MaNGA Maps."""

    route_base = '/maps/'

    def index(self):
        self.results['data'] = 'this is a maps!'
        return json.dumps(self.results)

    @flask.ext.classy.route('/<name>/', methods=['GET', 'POST'], endpoint='getMaps')
    def get(self, name):
        """Returns the parameters needed to initialise a Maps remotely.

        To initialise a Maps we need to return:
        - mangaid
        - plateifu
        - Header with WCS information
        - Maps shape
        - bintype
        - template_kin

        """

        maps, results = _getMaps(name)
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
            category (str):
                The category of the map to be extractred. E.g., `'EMLINE_GFLUX'`.
            channel (str or None):
                If the ``category`` contains multiple channels, the channel to use,
                e.g., ``Ha-6564'. Otherwise, ``None``.

        """

        name = kwargs.pop('name')
        category = kwargs.pop('category')
        channel = kwargs.pop('channel')

        # Initialises the Maps object
        maps, results = _getMaps(name, **kwargs)
        self.update_results(results)

        if maps is None:
            return json.dumps(self.results)

        try:
            mmap = maps.getMap(category=category, channel=channel)
            self.results['data'] = {}
            self.results['data']['value'] = mmap.value.tolist()
            self.results['data']['ivar'] = mmap.ivar.tolist()
            self.results['data']['mask'] = mmap.ivar.tolist()
            self.results['data']['unit'] = mmap.unit
        except Exception as ee:
            self.results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(ee))

        return json.dumps(self.results)
