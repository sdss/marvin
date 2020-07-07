# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-12-08 14:24:58
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-13 18:21:26

from __future__ import print_function, division, absolute_import
from flask_classful import FlaskView
from flask import request, current_app, session as current_session
from marvin.web.web_utils import parseSession, update_allowed, updateGlobalSession
from marvin.web.web_utils import check_access, check_request_for_release
import marvin
from brain.utils.general import build_routemap
from marvin.api.base import arg_validate as av


class BaseWebView(FlaskView):
    ''' This is the Base Web View for all pages '''

    def __init__(self, page):
        self.base = {}
        self.base['intro'] = 'Welcome to Marvin!'
        self.base['version'] = marvin.__version__
        self.update_title(page)
        self._endpoint = self._release = None
        self._drpver = self._dapver = None

    def before_request(self, *args, **kwargs):
        ''' this runs before every single request '''

        # check Flask request for release info but only when no session
        if 'release' not in current_session:
            check_request_for_release(request)

        # check login/access status and update global session
        check_access()
        updateGlobalSession()

        self.base['error'] = None
        self._logged_in = current_session.get('loginready', False)
        self._versions = update_allowed()
        self._endpoint = request.endpoint
        self._drpver, self._dapver, self._release = parseSession()

        # try to get a local version of the urlmap for the arg_validator
        if not av.urlmap:
            urlmap = build_routemap(current_app)
            marvin.config.urlmap = urlmap
            av.urlmap = urlmap

    def after_request(self, name, response):
        ''' this runs after every single request '''

        return response

    def update_title(self, page):
        ''' Update the title and page '''
        self.base['title'] = page.title().split('-')[0] if 'main' in page \
            else page.title().replace('-', ' | ')
        self.base['page'] = page

    def reset_dict(self, mydict, exclude=None):
        ''' resets the page dictionary '''
        mydict['error'] = self.base['error']
        exclude = exclude if isinstance(exclude, list) else [exclude]
        diffkeys = set(mydict) - set(self.base)
        for key, val in mydict.items():
            if key in diffkeys and (key not in exclude):
                mydict[key] = '' if isinstance(val, str) else None
        mydict['versions'] = self._versions
        mydict['release'] = self._release
        mydict['loggedin'] = self._logged_in
