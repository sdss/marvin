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
