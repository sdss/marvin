# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-12-08 14:24:58
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-02-08 14:53:24

from __future__ import print_function, division, absolute_import
from flask_classy import FlaskView
from flask import request


class BaseWebView(FlaskView):
    ''' This is the Base Web View for all pages '''

    def __init__(self):
        pass

    def before_request(self, *args, **kwargs):
        ''' this runs before every single request '''
        pass

    def after_request(self, name, response):
        ''' this runs after every single request '''

        return response

