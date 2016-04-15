#!/usr/bin/env python
# encoding: utf-8
#
# spaxel.py
#
# Licensed under a 3-clause BSD license.
#
# Revision history:
#     11 Apr 2016 J. Sánchez-Gallego
#       Initial version


from __future__ import division
from __future__ import print_function
from flask.ext.classy import route
from marvin.tools.spaxel import Spaxel
from marvin.api.base import BaseView
import json
from marvin.utils.general import parseName, parseRoutePath


def _getSpaxel(name, x, y):
    """Retrieves a Marvin Spaxel object."""

    spaxel = None
    results = {}

    # parse name into either mangaid or plateifu
    try:
        mangaid, plateifu = parseName(name)
    except Exception as e:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(e))
        return spaxel, results

    try:
        spaxel = Spaxel(x=x, y=y, mangaid=mangaid, plateifu=plateifu, mode='local')
        results['status'] = 1
    except Exception as e:
        results['error'] = 'Failed to retrieve Spaxels {0}: {1}'.format(name, str(e))

    return spaxel, results


class SpaxelView(BaseView):
    """Class describing API calls related to Spaxels."""

    route_base = '/spaxels/'

    @route('/<name>/<path:path>/', methods=['GET', 'POST'],
           endpoint='getSpaxel')
    @parseRoutePath
    def get(self, name, x=None, y=None):
        """Retrieves the spaxel at ``(x, y)`` and returns its arrays."""

        assert x is not None and y is not None

        spaxel, results = _getSpaxel(name, int(x), int(y))

        self.update_results(results)

        if spaxel is not None:
            self.results['data'] = {'data': spaxel.tolist(),
                                    'ivar': spaxel.ivar.tolist(),
                                    'mask': spaxel.mask.tolist(),
                                    'wavelength': spaxel.wavelength.tolist()}

        return json.dumps(self.results)
