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
import json

from flask.ext.classy import route
from flask import request

from marvin.api import parse_params
from marvin.tools.spaxel import Spaxel
from marvin.api.base import BaseView
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier
from brain.utils.general import parseRoutePath


def _getSpaxel(name, x, y):
    """Retrieves a Marvin Spaxel object."""

    spaxel = None
    results = {}

    drpver, __ = parse_params(request)

    # parse name into either mangaid or plateifu
    try:
        idtype = parseIdentifier(name)
    except Exception as e:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(e))
        return spaxel, results

    try:
        if idtype == 'plateifu':
            plateifu = name
            mangaid = None
        elif idtype == 'mangaid':
            mangaid = name
            plateifu = None
        else:
            raise MarvinError('invalid plateifu or mangaid: {0}'.format(idtype))

        spaxel = Spaxel(x=x, y=y, mangaid=mangaid, plateifu=plateifu, mode='local', drpver=drpver)
        results['status'] = 1
    except Exception as e:
        results['error'] = 'Failed to retrieve Spaxels {0}: {1}'.format(name, str(e))

    return spaxel, results


class SpaxelView(BaseView):
    """Class describing API calls related to Spaxels."""

    route_base = '/spaxels/'

    @route('/<name>/<path:path>/', methods=['GET', 'POST'], endpoint='getSpaxel')
    @parseRoutePath
    def get(self, name, x=None, y=None):
        """Retrieves the spaxel at ``(x, y)`` and returns its spectral data arrays."""

        assert x is not None and y is not None

        spaxel, results = _getSpaxel(name, int(x), int(y))

        self.update_results(results)

        if spaxel is not None:
            self.results['data'] = {'flux': spaxel.spectrum.flux.tolist(),
                                    'ivar': spaxel.spectrum.ivar.tolist(),
                                    'mask': spaxel.spectrum.mask.tolist(),
                                    'wavelength': spaxel.spectrum.wavelength.tolist(),
                                    'parent_shape': spaxel._parent_shape}

        return json.dumps(self.results)
