#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-04-28 14:07:58
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-04-28 14:07:58 by Brian Cherinka
    Last Modified On: 2016-04-28 14:07:58 by Brian

'''
from __future__ import print_function
from __future__ import division

from flask import current_app, Blueprint, render_template, session as current_session, request, redirect, url_for, jsonify
from flask.ext.classy import FlaskView, route
from brain.api.base import processRequest
from marvin.core import MarvinError
import os

plate = Blueprint("plate_page", __name__)


class Plate(FlaskView):
    route_base = '/plate'

    def __init__(self):
        ''' Initialize the route '''
        self.plate = {}
        self.plate['title'] = 'Marvin | Plate'
        self.plate['page'] = 'marvin-plate'
        self.plate['error'] = None
        self.plate['plateid'] = None

    def before_request(self, *args, **kwargs):
        ''' Do these things before a request to any route '''
        self.plate['plateid'] = None
        self.plate['error'] = None

    @route('/', methods=['GET', 'POST'])
    def index(self):

        # Attempt to retrieve search parameters
        form = processRequest(request=request)

        return render_template('plate.html', **self.plate)

    def get(self, plateid):
        ''' Retrieve info for a given plate id '''

        self.plate['plateid'] = plateid

        return render_template('plate.html', **self.plate)


Plate.register(plate)
