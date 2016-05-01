#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-04-29 01:15:33
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-04-29 01:15:33 by Brian Cherinka
    Last Modified On: 2016-04-29 01:15:33 by Brian

'''
from __future__ import print_function
from __future__ import division
from flask import current_app, Blueprint, render_template, session as current_session, request, redirect, url_for, jsonify
from flask.ext.classy import FlaskView, route
from brain.api.base import processRequest
from marvin.core import MarvinError
import os

images = Blueprint("images_page", __name__)


class Random(FlaskView):
    route_base = '/random'

    def __init__(self):
        ''' Initialize the route '''
        self.random = {}
        self.random['title'] = 'Marvin | Random'
        self.random['error'] = None

    def before_request(self, *args, **kwargs):
        ''' Do these things before a request to any route '''
        self.random['error'] = None

    @route('/', methods=['GET', 'POST'])
    def index(self):

        # Attempt to retrieve search parameters
        form = processRequest(request=request)
        self.random['imnumber'] = 16

        return render_template('random.html', **self.random)


Random.register(images)



