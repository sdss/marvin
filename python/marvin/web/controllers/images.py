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
from flask import Blueprint, render_template, request
from flask_classful import route
from brain.api.base import processRequest
from marvin.core.exceptions import MarvinError
from marvin.utils.general import getRandomImages
from marvin.web.web_utils import buildImageDict
from marvin.web.controllers import BaseWebView

images = Blueprint("images_page", __name__)


class Random(BaseWebView):
    route_base = '/random/'

    def __init__(self):
        ''' Initialize the route '''
        super(Random, self).__init__('marvin-random')
        self.random = self.base.copy()

    def before_request(self, *args, **kwargs):
        ''' Do these things before a request to any route '''
        super(Random, self).before_request(*args, **kwargs)
        self.reset_dict(self.random)

    @route('/', methods=['GET', 'POST'])
    def index(self):

        # Attempt to retrieve search parameters
        form = processRequest(request=request)
        self.random['imnumber'] = 16
        images = []

        # Get random images ; parse out thumbnails ; construct plate-IFUs
        imfiles = None
        try:
            imfiles = getRandomImages(as_url=True, num=self.random['imnumber'], mode='local', release=self._release)
        except (MarvinError, AssertionError) as e:
            self.random['error'] = 'Error: could not get images: {0}'.format(e)
        else:
            images = buildImageDict(imfiles)

        # if image grab failed, make placeholders
        if not imfiles:
            images = buildImageDict(imfiles, test=True, num=self.random['imnumber'])

        # Add images to dict
        self.random['images'] = images

        return render_template('random.html', **self.random)


Random.register(images)



