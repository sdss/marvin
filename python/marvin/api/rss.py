#!/usr/bin/env python
# encoding: utf-8
#
# rss.py
#
# Licensed under a 3-clause BSD license.
#
# Revision history:
#     11 Apr 2016 J. Sánchez-Gallego
#       Initial version


from __future__ import division
from __future__ import print_function
import json

import flask
from flask_classy import route

from marvin.api import parse_params
from marvin.tools.rss import RSS
from marvin.api.base import BaseView
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier


def _getRSS(name):
    """Retrieves a RSS Marvin object."""

    rss = None
    results = {}

    release = parse_params(flask.request)

    # parse name into either mangaid or plateifu
    try:
        idtype = parseIdentifier(name)
    except Exception as e:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(e))
        return rss, results

    try:
        if idtype == 'plateifu':
            plateifu = name
            mangaid = None
        elif idtype == 'mangaid':
            mangaid = name
            plateifu = None
        else:
            raise MarvinError('invalid plateifu or mangaid: {0}'.format(idtype))

        rss = RSS(mangaid=mangaid, plateifu=plateifu, mode='local', release=release)
        results['status'] = 1
    except Exception as e:
        results['error'] = 'Failed to retrieve RSS {0}: {1}'.format(name, str(e))

    return rss, results


class RSSView(BaseView):
    """Class describing API calls related to RSS files."""

    route_base = '/rss/'

    @route('/<name>/', methods=['GET', 'POST'], endpoint='getRSS')
    def get(self, name):
        """This method performs a get request at the url route /rss/<id>."""

        rss, results = _getRSS(name)
        self.update_results(results)

        if rss:
            # For now we don't return anything here, maybe later.
            self.results['data'] = {}

        return json.dumps(self.results)

    @route('/<name>/fibers/', methods=['GET', 'POST'], endpoint='getRSSAllFibers')
    def getAllFibers(self, name):
        """Returns a list of all the flux, ivar, mask, and wavelength arrays for all fibres."""

        rss, results = _getRSS(name)
        self.update_results(results)

        if rss:
            self.results['data'] = {}
            self.results['data']['wavelength'] = rss[0].wavelength.tolist()

            for ii, fiber in enumerate(rss):
                flux = fiber.flux.tolist()
                ivar = fiber.ivar.tolist()
                mask = fiber.mask.tolist()
                self.results['data'][ii] = [flux, ivar, mask]

        return json.dumps(self.results)
