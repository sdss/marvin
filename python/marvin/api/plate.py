#!/usr/bin/env python
# encoding: utf-8


# Created by Brian Cherinka on 2016-05-18 15:08:29
# Licensed under a 3-clause BSD license.

# Revision History:
#     Initial Version: 2016-05-18 15:08:29 by Brian Cherinka
#     Last Modified On: 2016-05-18 15:08:29 by Brian


from __future__ import print_function
from __future__ import division
import json
from flask_classy import route
from marvin.tools.plate import Plate
from marvin.api.base import BaseView
from marvin.core.exceptions import MarvinError


def _getPlate(plateid, nocubes=None):
    ''' Get a Plate Marvin Object '''
    plate = None
    results = {}

    if not str(plateid).isdigit():
        results['error'] = 'Error: plateid is not a numeric value'
        return plate, results

    try:
        plate = Plate(plateid=plateid, nocubes=nocubes, mode='local')
    except Exception as e:
        results['error'] = 'Failed to retrieve Plate for id {0}: {1}'.format(plateid, str(e))
    else:
        results['status'] = 1

    return plate, results


class PlateView(BaseView):
    """Class describing API calls related to plates."""

    route_base = '/plate/'

    @route('/<plateid>/', methods=['GET', 'POST'], endpoint='getPlate')
    def get(self, plateid):
        """This method performs a get request at the url route /plate/<id>."""

        plate, results = _getPlate(plateid, nocubes=True)
        self.update_results(results)

        if not isinstance(plate, type(None)):
            # For now we don't return anything here, maybe later.

            platedict = {'plateid': plateid, 'header': plate._hdr}
            self.results['data'] = platedict

        return json.dumps(self.results)

    @route('/<plateid>/cubes/', methods=['GET', 'POST'], endpoint='getPlateCubes')
    def getPlateCubes(self, plateid):
        """Returns a list of all the cubes for this plate """

        plate, results = _getPlate(plateid)
        self.update_results(results)

        if not isinstance(plate, type(None)):
            plateifus = [cube.plateifu for cube in plate]
            self.results['data'] = {'plateifus': plateifus}

        return json.dumps(self.results)
