#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2016-04-11
# @Filename: rss.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-08-04 14:06:38


from __future__ import division, print_function

import io
import os
import sys

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
                filename = Path(release=release).full('mangarss', ifu=ifu, plate=plate, drpver=drpver)
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

            obsinfo = io.StringIO() if sys.version_info.major >= 3 else io.BytesIO()
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

    @route('/<name>/fibers/<fiberid>', methods=['GET', 'POST'], endpoint='getRSSFiber')
    @av.check_args()
    def getFiber(self, args, name, fiberid):
        """Returns a list of all the RSS arrays for a given fibre.

        .. :quickref: RSS; Get a list of all the RSS arrays for a given fibre.

        :param name: The name of the cube as plate-ifu or mangaid
        :param fiberid: The fiberid of the fibre to retrieve.
        :form release: the release of MaNGA
        :resjson int status: status of response. 1 if good, -1 if bad.
        :resjson string error: error message, null if None
        :resjson json inconfig: json of incoming configuration
        :resjson json utahconfig: json of outcoming configuration
        :resjson string traceback: traceback of an error, null if None
        :resjson json data: dictionary of returned data
        :resheader Content-Type: application/json
        :statuscode 200: no error
        :statuscode 422: invalid input parameters

        **Example request**:

        .. sourcecode:: http

           GET /marvin/api/rss/8485-1901/fibers/15 HTTP/1.1
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
              "data": {"flux": [1., 2., 3., ...]
                       "wavelength": [3621.6, 3622.43, 3623.26, ...],
                       "ivar: ...,
                       "mask: ...,
                       "dispersion": ...
                       ...
              }
           }

        """

        # Pop any args we don't want going into Rss
        args = self._pop_args(args, arglist='name')

        rss, res = _getRSS(name, **args)
        self.update_results(res)

        if rss:

            self.results['data'] = {}

            for ext in rss.data:
                if ext.data is None or ext.name == 'OBSINFO':
                    continue
                if ext.data.ndim == 2:
                    self.results['data'][ext.name] = ext.data[int(fiberid), :].tolist()
                else:
                    self.results['data'][ext.name] = ext.data.tolist()

        return jsonify(self.results)
