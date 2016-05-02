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
from collections import defaultdict
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
        images = []
        imdict = defaultdict(str)
        # test run
        for i in xrange(self.random['imnumber']):
            imdict['name'] = '8485-1901'
            imdict['image'] = 'http://localhost:80/sas/mangawork/manga/spectro/redux/v1_5_1/8485/stack/images/1901.png'
            imdict['thumb'] = 'http://localhost:80/sas/mangawork/manga/spectro/redux/v1_5_1/8485/stack/images/1901_thumb.png'
            images.append(imdict)
        self.random['images'] = images

        return render_template('random.html', **self.random)


Random.register(images)



