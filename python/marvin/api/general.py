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

import json
from flask_classy import route

from brain.api.general import BrainGeneralRequestsView
from marvin.utils.general import mangaid2plateifu as mangaid2plateifu
from marvin.utils.general import get_nsa_data


class GeneralRequestsView(BrainGeneralRequestsView):

    @route('/mangaid2plateifu/<mangaid>/', endpoint='mangaid2plateifu', methods=['GET', 'POST'])
    def mangaid2plateifu(self, mangaid):

        try:
            plateifu = mangaid2plateifu(mangaid, mode='db')
            self.results['data'] = plateifu
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = ('manga2plateifu failed with error: {0}'.format(str(ee)))

        return json.dumps(self.results)

    @route('/nsa/full/<mangaid>/', endpoint='nsa_full', methods=['GET', 'POST'])
    def get_nsa_data(self, mangaid):
        """Returns the NSA data for a given mangaid from the full catalogue."""

        try:
            nsa_data = get_nsa_data(mangaid, mode='local', source='nsa')
            self.results['data'] = nsa_data
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = 'get_nsa_data failed with error: {0}'.format(str(ee))

        return json.dumps(self.results)

    @route('/nsa/drpall/<mangaid>/', endpoint='nsa_drpall', methods=['GET', 'POST'])
    def get_nsa_drpall_data(self, mangaid):
        """Returns the NSA data in drpall for a given mangaid.

        Note that this always uses the drpver/drpall versions that are default in the server.

        """

        try:
            nsa_data = get_nsa_data(mangaid, mode='local', source='drpall')
            self.results['data'] = nsa_data
            self.results['status'] = 1
        except Exception as ee:
            self.results['status'] = -1
            self.results['error'] = 'get_nsa_data failed with error: {0}'.format(str(ee))

        return json.dumps(self.results)
