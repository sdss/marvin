#!/usr/bin/env python
# encoding: utf-8
"""

general.py

Licensed under a 3-clause BSD license.

Revision history:
    17 Feb 2016 J. Sánchez-Gallego
      Initial version

"""

from __future__ import division
from __future__ import print_function
from flask.ext.classy import route
from marvin.api.base import BaseView
from marvin.utils.general import mangaid2plateifu as mangaid2plateifu
#from marvin.api.cube import api
from flask import Blueprint
import json


#apiGeneral = Blueprint("apiGeneral", __name__)


class GeneralRequestsView(BaseView):
    """A collection of requests for generic purposes."""

    route_base = '/api/general/'

    @route('/mangaid2plateifu/<mangaid>')
    def mangaid2plateifu(self, mangaid):

        #results = {'data': None, 'error': None, 'status': None}

        try:
            plateifu = mangaid2plateifu(mangaid, mode='db')
            self.results['data'] = plateifu
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = ('manga2plateifu failed with error: {0}'.format(str(ee)))

        return json.dumps(self.results)


#GeneralRequestsView.register(apiGeneral)
