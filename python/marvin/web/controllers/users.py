# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-09-28 16:21:17
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2016-10-11 20:07:56

from __future__ import print_function, division, absolute_import
from flask import current_app, Blueprint, render_template, jsonify
from flask import session as current_session, request, redirect, url_for
from flask_classy import FlaskView, route
from marvin import config, marvindb
from brain.api.base import processRequest
from marvin.core.exceptions import MarvinError
from marvin.utils.general.general import parseIdentifier
from wtforms import SelectField, validators
import json

users = Blueprint("users_page", __name__)


class User(FlaskView):
    route_base = '/user'

    def __init__(self):
        ''' Initialize the route '''
        self.user = {}
        self.user['title'] = 'Marvin | User'
        self.user['page'] = 'marvin-user'
        self.user['error'] = None

    def index(self):
        pass

    def preferences(self):
        return render_template('preferences.html', **self.user)

User.register(users)
