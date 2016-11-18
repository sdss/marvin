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
from flask_classy import route
from marvin.api.base import BaseView
from marvin.utils.general import mangaid2plateifu as mangaid2plateifu
import json


class GeneralRequestsView(BaseView):

    route_base = '/mangaid2plateifu/'

    @route('/<mangaid>/', endpoint='mangaid2plateifu')
    def mangaid2plateifu(self, mangaid):

        try:
            plateifu = mangaid2plateifu(mangaid, mode='db')
            self.results['data'] = plateifu
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = ('manga2plateifu failed with error: {0}'.format(str(ee)))

        return json.dumps(self.results)
