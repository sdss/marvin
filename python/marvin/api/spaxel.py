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

from flask_classy import route
from flask import request

from marvin.api import parse_params
from marvin.tools.spaxel import Spaxel
from marvin.api.base import BaseView
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier


def _getSpaxel(name, x, y, **kwargs):
    """Retrieves a Marvin Spaxel object."""

    spaxel = None
    results = {}

    release = parse_params(request)

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

        spaxel = Spaxel(x=x, y=y, mangaid=mangaid, plateifu=plateifu,
                        release=release, **kwargs)
        results['status'] = 1
    except Exception as e:
        results['error'] = 'Failed to retrieve Spaxels {0}: {1}'.format(name, str(e))

    return spaxel, results


class SpaxelView(BaseView):
    """Class describing API calls related to Spaxels."""

    route_base = '/spaxels/'

    @route('/<name>/spectra/<x>/<y>/', methods=['GET', 'POST'], endpoint='getSpectrum')
    def spectrum(self, name, x, y):
        """Returns a dictionary with the DRP spectrum for a spaxel.

        Loads a DRP Cube and uses getSpaxel to retrieve the ``(x,y)``
        spaxel. Returns a dictionary with the spectrum for that spaxel.

        Parameters:
            name (str):
                The ``plateifu`` or ``mangaid`` of the object.
            x,y (int):
                The x/y coordinates of the spaxel (origin is ``lower``).

        """

        spaxel, results = _getSpaxel(name, int(x), int(y),
                                     maps=False, modelcube=False)

        self.update_results(results)

        if spaxel is not None:
            self.results['data'] = {'flux': spaxel.spectrum.flux.tolist(),
                                    'ivar': spaxel.spectrum.ivar.tolist(),
                                    'mask': spaxel.spectrum.mask.tolist(),
                                    'wavelength': spaxel.spectrum.wavelength.tolist(),
                                    'specres': spaxel.specres.tolist()}

        return json.dumps(self.results)

    @route('/<name>/properties/<template_kin>/<x>/<y>/',
           methods=['GET', 'POST'], endpoint='getProperties')
    def properties(self, name, x, y, template_kin):
        """Returns a dictionary with the DAP properties for a spaxel.

        Loads a DAP Maps and uses getSpaxel to retrieve the ``(x,y)``
        spaxel. Returns a dictionary with the properties for that spaxel.

        Parameters:
            name (str):
                The ``plateifu`` or ``mangaid`` of the object.
            x,y (int):
                The x/y coordinates of the spaxel (origin is ``lower``).
            template_kin (str):
                The template_kin associated with this model cube.
                If not defined, the default template_kin will be used.

        """

        spaxel, results = _getSpaxel(name, int(x), int(y),
                                     template_kin=template_kin,
                                     cube=False, modelcube=False)

        self.update_results(results)

        if spaxel is not None:
            spaxel_properties = {}
            for name in spaxel.properties:
                prop = spaxel.properties[name]
                spaxel_properties[name] = {}
                for key in ['name', 'channel', 'value', 'ivar', 'mask', 'description', 'unit']:
                    spaxel_properties[name][key] = getattr(prop, key)

            self.results['data'] = {'properties': spaxel_properties}

        return json.dumps(self.results)

    @route('/<name>/models/<template_kin>/<x>/<y>/',
           methods=['GET', 'POST'], endpoint='getModels')
    def getModels(self, name, x, y, template_kin):
        """Returns a dictionary with the models for a spaxel.

        Loads a ModelCube and uses getSpaxel to retrieve the ``(x,y)``
        spaxel. Returns a dictionary with the models for that spaxel.

        Parameters:
            name (str):
                The ``plateifu`` or ``mangaid`` of the object.
            x,y (int):
                The x/y coordinates of the spaxel (origin is ``lower``).
            template_kin (str):
                The template_kin associated with this model cube.
                If not defined, the default template_kin will be used.

        """

        spaxel, results = _getSpaxel(name, x, y,
                                     template_kin=template_kin,
                                     cube=False, maps=False)

        self.update_results(results)

        if spaxel is not None:

            self.results['data'] = {
                'flux_array': spaxel.model_flux.flux.tolist(),
                'flux_ivar': spaxel.model_flux.ivar.tolist(),
                'flux_mask': spaxel.model_flux.mask.tolist(),
                'model_array': spaxel.model.flux.tolist(),
                'model_emline': spaxel.emline.flux.tolist(),
                'model_emline_base': spaxel.emline_base.flux.tolist(),
                'model_emline_mask': spaxel.emline.mask.tolist(),
                'bintype': spaxel.bintype,
                'template_kin': spaxel.template_kin}

        return json.dumps(self.results)
