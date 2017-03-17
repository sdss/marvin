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

from flask import jsonify
from flask_classy import route

from marvin.tools.rss import RSS
from marvin.api.base import BaseView, arg_validate as av
from marvin.core.exceptions import MarvinError
from marvin.utils.general import parseIdentifier


def _getRSS(name, **kwargs):
    """Retrieves a RSS Marvin object."""

    rss = None
    results = {}

    # Pop the release to remove a duplicate input to Maps
    release = kwargs.pop('release', None)

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
    @av.check_args()
    def get(self, args, name):
        """This method performs a get request at the url route /rss/<id>.

        .. :quickref: RSS; Get an RSS given a plate-ifu or mangaid

        :param name: The name of the cube as plate-ifu or mangaid
        :form release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json string empty: the data dict is empty
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/rss/8485-1901/ HTTP/1.1
           Host: api.sdss.org
           Accept: application/json, */*

        **Example response**:

        .. sourcecode:: http

           HTTP/1.1 200 OK
           Content-Type: application/json
           {
              "status": 1,
              "error": null,
              "inconfig": {"release": "MPL-5"},
              "utahconfig": {"release": "MPL-5", "mode": "local"},
              "traceback": null,
              "data": {}
           }

        """

        # Pop any args we don't want going into Rss
        args = self._pop_args(args, arglist='name')

        rss, results = _getRSS(name, **args)
        self.update_results(results)

        if rss:
            # For now we don't return anything here, maybe later.
            self.results['data'] = {}

        return jsonify(self.results)

    @route('/<name>/fibers/', methods=['GET', 'POST'], endpoint='getRSSAllFibers')
    @av.check_args()
    def getAllFibers(self, args, name):
        """Returns a list of all the flux, ivar, mask, and wavelength arrays for all fibres.

        .. :quickref: RSS; Get a list of flux, ivar, mask, and wavelength arrays for all fibers

        :param name: The name of the cube as plate-ifu or mangaid
        :form release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :json list rssfiber: the flux, ivar, mask arrays for the given rssfiber index
        :json list wavelength: the wavelength arrays for all fibers
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin2/api/rss/8485-1901/fibers/ HTTP/1.1
           Host: api.sdss.org
           Accept: application/json, */*

        **Example response**:

        .. sourcecode:: http

           HTTP/1.1 200 OK
           Content-Type: application/json
           {
              "status": 1,
              "error": null,
              "inconfig": {"release": "MPL-5"},
              "utahconfig": {"release": "MPL-5", "mode": "local"},
              "traceback": null,
              "data": {"wavelength": [3621.6, 3622.43, 3623.26, ...],
                       "0": [flux, ivar, mask],
                       "1": [flux, ivar, mask],
                       ...
                       "170": [flux, ivar, mask]
              }
           }

        """

        # Pop any args we don't want going into Rss
        args = self._pop_args(args, arglist='name')

        rss, results = _getRSS(name, **args)
        self.update_results(results)

        if rss:
            self.results['data'] = {}
            self.results['data']['wavelength'] = rss[0].wavelength.tolist()

            for ii, fiber in enumerate(rss):
                flux = fiber.flux.tolist()
                ivar = fiber.ivar.tolist()
                mask = fiber.mask.tolist()
                self.results['data'][ii] = [flux, ivar, mask]

        return jsonify(self.results)
