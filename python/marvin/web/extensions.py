# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-28 15:31:51
# @Last modified by: Brian Cherinka
# @Last Modified time: 2018-06-05 11:39:50

from __future__ import absolute_import, division, print_function

import logging

import flask_jsglue as jsg
from flask_caching import Cache
from flask_featureflags import FeatureFlag
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_profiler import Profiler
from raven.contrib.flask import Sentry


# JS Glue (allows use of Flask.url_for inside javascript)
jsglue = jsg.JSGlue()

# Feature Flags (allows turning on/off of features)
flags = FeatureFlag()

# Sentry error logging
sentry = Sentry(logging=True, level=logging.ERROR)

# Route Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# Flask Profiler
profiler = Profiler()

# Flask Cache
cache = Cache()

# Flask Login
login_manager = LoginManager()

# Flask-JWT (JSON Web Token)
jwt = JWTManager()
