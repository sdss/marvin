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


# from __future__ import division
# from __future__ import print_function
#
# import numpy as np
#
# from flask_classful import route
# from flask import jsonify
#
# from marvin.tools.spaxel import Spaxel
# from marvin.api.base import BaseView, arg_validate as av
# from marvin.core.exceptions import MarvinError
# from marvin.utils.general import parseIdentifier
#
#
# def _getSpaxel(name, x, y, **kwargs):
#     """Retrieves a Marvin Spaxel object."""
#
#     spaxel = None
#     results = {}
#
#     # Pop the release to remove a duplicate input to Maps
#     release = kwargs.pop('release', None)
#
#     # parse name into either mangaid or plateifu
#     try:
#         idtype = parseIdentifier(name)
#     except Exception as e:
#         results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(e))
#         return spaxel, results
#
#     try:
#         if idtype == 'plateifu':
#             plateifu = name
#             mangaid = None
#         elif idtype == 'mangaid':
#             mangaid = name
#             plateifu = None
#         else:
#             raise MarvinError('invalid plateifu or mangaid: {0}'.format(idtype))
#
#         spaxel = Spaxel(x=x, y=y, mangaid=mangaid, plateifu=plateifu,
#                         release=release, **kwargs)
#         results['status'] = 1
#     except Exception as e:
#         results['error'] = 'Failed to retrieve Spaxels {0}: {1}'.format(name, str(e))
#
#     return spaxel, results
#
#
# class SpaxelView(BaseView):
#     """Class describing API calls related to Spaxels."""
#
#     route_base = '/spaxels/'
#
#     @route('/<name>/spectra/<x>/<y>/', methods=['GET', 'POST'], endpoint='getSpectrum')
#     @av.check_args()
#     def spectrum(self, args, name, x, y):
#         """Returns a dictionary with the DRP spectrum for a spaxel.
#
#         Loads a DRP Cube and uses getSpaxel to retrieve the ``(x,y)``
#         spaxel. Returns a dictionary with the spectrum for that spaxel.
#
#         .. :quickref: Spaxel; Get a spectrum from a specific spaxel from a DRP cube
#
#         :param name: The name of the object as plate-ifu or mangaid
#         :param x: The x coordinate of the spaxel (origin is ``lower``)
#         :param y: The y coordinate of the spaxel (origin is ``lower``)
#         :form inconfig: json of any incoming parameters
#         :resjson int status: status of response. 1 if good, -1 if bad.
#         :resjson string error: error message, null if None
#         :resjson json inconfig: json of incoming configuration
#         :resjson json utahconfig: json of outcoming configuration
#         :resjson string traceback: traceback of an error, null if None
#         :resjson json data: dictionary of returned data
#         :json list flux: the spectrum flux array
#         :json list ivar: the spectrum ivar array
#         :json list mask: the spectrum mask array
#         :json list wavelength: the spectrum wavelength array
#         :json list specres: the spectrum spectral resolution array
#         :resheader Content-Type: application/json
#         :statuscode 200: no error
#         :statuscode 422: invalid input parameters
#
#         **Example request**:
#
#         .. sourcecode:: http
#
#            GET /marvin/api/spaxels/8485-1901/spectra/10/10/ HTTP/1.1
#            Host: api.sdss.org
#            Accept: application/json, */*
#
#         **Example response**:
#
#         .. sourcecode:: http
#
#            HTTP/1.1 200 OK
#            Content-Type: application/json
#            {
#               "status": 1,
#               "error": null,
#               "inconfig": {"release": "MPL-5"},
#               "utahconfig": {"release": "MPL-5", "mode": "local"},
#               "traceback": null,
#               "data": {"flux": [-0.001416, 0.0099, 0.0144, ...],
#                     "ivar": [134.613, 133.393, 132.094, ...],
#                     "mask": [0, 0, 0, ...],
#                     "wavelength": [3621.6, 3622.43, ..., 10353.8],
#                     "specres": [1026.83, 1027.07, 1027.3]
#               }
#            }
#
#         """
#
#         # Pop any args we don't want going into Spaxel
#         args = self._pop_args(args, arglist=['name', 'x', 'y'])
#
#         spaxel, results = _getSpaxel(name, x, y, maps=False, modelcube=False, **args)
#
#         self.update_results(results)
#
#         if spaxel is not None:
#             self.results['data'] = {'flux': spaxel.spectrum.value.tolist(),
#                                     'ivar': spaxel.spectrum.ivar.tolist(),
#                                     'mask': spaxel.spectrum.mask.tolist(),
#                                     'wavelength': spaxel.spectrum.wavelength.value.tolist(),
#                                     'specres': spaxel.specres.tolist()}
#
#         return jsonify(self.results)
#
#     @route('/<name>/properties/<template>/<x>/<y>/',
#            methods=['GET', 'POST'], endpoint='getProperties')
#     @av.check_args()
#     def properties(self, args, name, x, y, template):
#         """Returns a dictionary with the DAP properties for a spaxel.
#
#         Loads a DAP Maps and uses getSpaxel to retrieve the ``(x,y)``
#         spaxel. Returns a dictionary with the properties for that spaxel.
#
#         .. :quickref: Spaxel; Get DAP properties from a specific spaxel from a DAP Maps
#
#         :param name: The name of the object as plate-ifu or mangaid
#         :param x: The x coordinate of the spaxel (origin is ``lower``)
#         :param y: The y coordinate of the spaxel (origin is ``lower``)
#         :param template: The template associated with this maps. If none, default is used.
#         :form inconfig: json of any incoming parameters
#         :resjson int status: status of response. 1 if good, -1 if bad.
#         :resjson string error: error message, null if None
#         :resjson json inconfig: json of incoming configuration
#         :resjson json utahconfig: json of outcoming configuration
#         :resjson string traceback: traceback of an error, null if None
#         :resjson json data: dictionary of returned data
#         :json dict properties: the DAP properties for this spaxel
#         :resheader Content-Type: application/json
#         :statuscode 200: no error
#         :statuscode 422: invalid input parameters
#
#         **Example request**:
#
#         .. sourcecode:: http
#
#            GET /marvin/api/spaxels/8485-1901/properties/GAU-MILESHC/10/10/ HTTP/1.1
#            Host: api.sdss.org
#            Accept: application/json, */*
#
#         **Example response**:
#
#         .. sourcecode:: http
#
#            HTTP/1.1 200 OK
#            Content-Type: application/json
#            {
#               "status": 1,
#               "error": null,
#               "inconfig": {"release": "MPL-5"},
#               "utahconfig": {"release": "MPL-5", "mode": "local"},
#               "traceback": null,
#               "data": {"properties": {
#                           "bin_area": {
#                             "channel": null,
#                             "description": "Area of each bin.",
#                             "ivar": null,
#                             "mask": null,
#                             "name": "bin_area",
#                             "unit": "arcsec^2",
#                             "value": 0.5
#                           },
#                           ...
#                        }
#               }
#            }
#
#         """
#
#         # Pop any args we don't want going into Spaxel
#         args = self._pop_args(args, arglist=['name', 'x', 'y'])
#
#         spaxel, results = _getSpaxel(name, x, y, cube=False, modelcube=False, **args)
#
#         self.update_results(results)
#
#         if spaxel is not None:
#             spaxel_properties = {}
#             for name in spaxel.properties:
#                 prop = spaxel.properties[name]
#                 spaxel_properties[name] = {}
#                 for key in ['name', 'value', 'ivar', 'mask', 'description']:
#                     propval = getattr(prop, key)
#                     if type(propval).__module__ == np.__name__:
#                         propval = np.asscalar(str(propval))
#                     spaxel_properties[name][key] = propval
#                 spaxel_properties[name]['channel'] = prop.channel.name if prop.channel else None
#                 spaxel_properties[name]['unit'] = prop.unit.to_string()
#
#             self.results['data'] = {'properties': spaxel_properties}
#
#         return jsonify(self.results)
#
#     @route('/<name>/models/<template>/<x>/<y>/',
#            methods=['GET', 'POST'], endpoint='getModels')
#     @av.check_args()
#     def getModels(self, args, name, x, y, template):
#         """Returns a dictionary with the models for a spaxel.
#
#         Loads a ModelCube and uses getSpaxel to retrieve the ``(x,y)``
#         spaxel. Returns a dictionary with the models for that spaxel.
#
#         .. :quickref: Spaxel; Get the models for a specific spaxel from a DAP ModelCube
#
#         :param name: The name of the object as plate-ifu or mangaid
#         :param x: The x coordinate of the spaxel (origin is ``lower``)
#         :param y: The y coordinate of the spaxel (origin is ``lower``)
#         :param template: The template associated with this maps. If none, default is used.
#         :form inconfig: json of any incoming parameters
#         :resjson int status: status of response. 1 if good, -1 if bad.
#         :resjson string error: error message, null if None
#         :resjson json inconfig: json of incoming configuration
#         :resjson json utahconfig: json of outcoming configuration
#         :resjson string traceback: traceback of an error, null if None
#         :resjson json data: dictionary of returned data
#         :json list flux_array: flux of the binned spectrum
#         :json list flux_ivar: ivar of the binned spectrum
#         :json list flux_mask: mask of the binned spectrum and model
#         :json list model_array: best fitting model spectra
#         :json list model_emline: model spectrum with only emission lines
#         :json list model_emline_base: model of constant baseline fitted beneath emission lines
#         :json list model_emline_mask: bitmask that applies only to emission-line modeling
#         :json string bintype: the spectrum spectral resolution array
#         :json string template: the spectrum spectral resolution array
#         :resheader Content-Type: application/json
#         :statuscode 200: no error
#         :statuscode 422: invalid input parameters
#
#         **Example request**:
#
#         .. sourcecode:: http
#
#            GET /marvin/api/spaxels/8485-1901/models/GAU-MILESHC/10/10/ HTTP/1.1
#            Host: api.sdss.org
#            Accept: application/json, */*
#
#         **Example response**:
#
#         .. sourcecode:: http
#
#            HTTP/1.1 200 OK
#            Content-Type: application/json
#            {
#               "status": 1,
#               "error": null,
#               "inconfig": {"release": "MPL-5"},
#               "utahconfig": {"release": "MPL-5", "mode": "local"},
#               "traceback": null,
#               "data": {"bintype": "SPX",
#                     "template": "GAU-MILESHC",
#                     "flux_array": [-0.001416, 0.0099, 0.0144, ...],
#                     "flux_ivar": [134.613, 133.393, 132.094, ...],
#                     "flux_mask": [32, 32, 32, ...],
#                     "model_array": [0, 0, 0, ...],
#                     "model_emline": [0, 0, 0, ...],
#                     "model_emline_base": [0, 0, 0, ...],
#                     "model_emline_mask": [128, 128, 128, ...],
#               }
#            }
#
#         """
#
#         # Pop any args we don't want going into Spaxel
#         args = self._pop_args(args, arglist=['name', 'x', 'y'])
#
#         spaxel, results = _getSpaxel(name, x, y, cube=False, maps=False, **args)
#
#         self.update_results(results)
#
#         if spaxel is not None:
#
#             self.results['data'] = {
#                 'flux_array': spaxel.model_flux.value.tolist(),
#                 'flux_ivar': spaxel.model_flux.ivar.tolist(),
#                 'flux_mask': spaxel.model_flux.mask.tolist(),
#                 'model_array': spaxel.model.value.tolist(),
#                 'model_emline': spaxel.emline.value.tolist(),
#                 'model_emline_base': spaxel.emline_base.value.tolist(),
#                 'model_emline_mask': spaxel.emline.mask.tolist(),
#                 'bintype': spaxel.bintype.name,
#                 'template': spaxel.template.name}
#
#         return jsonify(self.results)
