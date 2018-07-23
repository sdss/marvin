#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2016-04-11
# @Filename: rss.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-23 01:06:18


from __future__ import division, print_function

import io
import os

import numpy
from brain.core.exceptions import BrainError
from flask import jsonify
from flask_classful import route
from sdss_access.path import Path

import marvin
from marvin.api.base import BaseView
from marvin.api.base import arg_validate as av
from marvin.core.exceptions import MarvinError
from marvin.utils.general import mangaid2plateifu, parseIdentifier


def _getRSS(name, use_file=True, release=None, **kwargs):
    """Retrieves a RSS Marvin object."""

    drpver, __ = marvin.config.lookUpVersions(release)

    rss = None
    results = {}

    # parse name into either mangaid or plateifu
    try:
        idtype = parseIdentifier(name)
    except Exception as ee:
        results['error'] = 'Failed to parse input name {0}: {1}'.format(name, str(ee))
        return rss, results

    filename = None
    plateifu = None
    mangaid = None

    try:
        if use_file:

            if idtype == 'mangaid':
                plate, ifu = mangaid2plateifu(name, drpver=drpver)
            elif idtype == 'plateifu':
                plate, ifu = name.split('-')

            if Path is not None:
                filename = Path().full('mangarss', ifu=ifu, plate=plate, drpver=drpver)
                assert os.path.exists(filename), 'file not found.'
            else:
                raise MarvinError('cannot create path for MaNGA rss.')

        else:

            if idtype == 'plateifu':
                plateifu = name
            elif idtype == 'mangaid':
                mangaid = name
            else:
                raise MarvinError('invalid plateifu or mangaid: {0}'.format(idtype))

        rss = marvin.tools.RSS(filename=filename, mangaid=mangaid, plateifu=plateifu,
                               mode='local', release=release)

        results['status'] = 1

    except Exception as ee:

        results['error'] = 'Failed to retrieve RSS {0}: {1}'.format(name, str(ee))

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

           GET /marvin/api/rss/8485-1901/ HTTP/1.1
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

        rss, res = _getRSS(name, **args)
        self.update_results(res)

        if rss:

            try:
                nsa_data = rss.nsa
            except (MarvinError, BrainError):
                nsa_data = None

            wavelength = (rss._wavelength.tolist() if isinstance(rss._wavelength, numpy.ndarray)
                          else rss._wavelength)

            obsinfo = io.StringIO()
            rss.obsinfo.write(format='ascii', filename=obsinfo)
            obsinfo.seek(0)

            self.results['data'] = {'plateifu': name,
                                    'mangaid': rss.mangaid,
                                    'ra': rss.ra,
                                    'dec': rss.dec,
                                    'header': rss.header.tostring(),
                                    'redshift': nsa_data.z if nsa_data else -9999,
                                    'wavelength': wavelength,
                                    'wcs_header': rss.wcs.to_header_string(),
                                    'nfibers': rss._nfibers,
                                    'obsinfo': obsinfo.read()}

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

           GET /marvin/api/rss/8485-1901/fibers/ HTTP/1.1
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
            self.results['data']['wavelength'] = rss[0].wavelength.value.tolist()

            for ii, fiber in enumerate(rss):
                flux = fiber.value.tolist()
                ivar = fiber.ivar.tolist()
                mask = fiber.mask.tolist()
                self.results['data'][ii] = [flux, ivar, mask]

        return jsonify(self.results)
