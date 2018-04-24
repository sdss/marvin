# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-09-28 16:21:17
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-04-24 13:26:54

from __future__ import print_function, division, absolute_import
from flask import Blueprint, render_template
from marvin.web.controllers import BaseWebView


users = Blueprint("users_page", __name__)


class User(BaseWebView):
    route_base = '/user/'

    def __init__(self):
        ''' Initialize the route '''
        super(User, self).__init__('marvin-user')
        self.user = self.base.copy()

    def index(self):
        return render_template('preferences.html', **self.user)

    def preferences(self):
        return render_template('preferences.html', **self.user)

User.register(users)
