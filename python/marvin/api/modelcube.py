#!/usr/bin/env python
# encoding: utf-8
#
# modelcube.py
#
# Licensed under a 3-clause BSD license.
#
# Revision history:
#     25 Sep 2016 J. Sánchez-Gallego
#       Initial version


from __future__ import division
from __future__ import print_function
import json

from flask_classy import route
from flask import request

from marvin.api import parse_params
from marvin.tools.modelcube import ModelCube
from marvin.api.base import BaseView
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier


def _get_model_cube(name, **kwargs):
    """Retrieves a Marvin ModelCube object."""

    model_cube = None
    results = {}

    release = parse_params(request)

    # parse name into either mangaid or plateifu
    try:
        idtype = parseIdentifier(name)
    except Exception as err:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(err))
        return model_cube, results

    try:
        if idtype == 'plateifu':
            plateifu = name
            mangaid = None
        elif idtype == 'mangaid':
            mangaid = name
            plateifu = None
        else:
            raise MarvinError('invalid plateifu or mangaid: {0}'.format(idtype))

        model_cube = ModelCube(mangaid=mangaid, plateifu=plateifu,
                               release=release, **kwargs)
        results['status'] = 1
    except Exception as err:
        results['error'] = 'Failed to retrieve ModelCube {0}: {1}'.format(name, str(err))

    return model_cube, results


class ModelCubeView(BaseView):
    """Class describing API calls related to ModelCubes."""

    route_base = '/modelcubes/'

    @route('/<name>/<bintype>/<template_kin>/', methods=['GET', 'POST'], endpoint='getModelCube')
    def get(self, name, bintype, template_kin):
        """Retrieves a ModelCube.

        Parameters:
            name (str):
                The ``plateifu`` or ``mangaid`` of the object.
            bintype (str):
                The bintype associated with this model cube. If not defined,
                the default type of binning will be used.
            template_kin (str):
                The template_kin associated with this model cube.
                If not defined, the default template_kin will be used.

        """

        model_cube, results = _get_model_cube(name, bintype=bintype, template_kin=template_kin)

        self.update_results(results)

        if model_cube is not None:
            self.results['data'] = {
                'header': model_cube.header.tostring(),
                'shape': model_cube.shape,
                'wavelength': model_cube.wavelength.tolist(),
                'redcorr': model_cube.redcorr.tolist(),
                'wcs_header': model_cube.wcs.to_header_string(),
                'bintype': model_cube.bintype,
                'template_kin': model_cube.template_kin}

        return json.dumps(self.results)
