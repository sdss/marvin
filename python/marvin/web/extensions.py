# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-28 15:31:51
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-06-28 18:14:12

from __future__ import print_function, division, absolute_import
from flask_featureflags import FeatureFlag
from raven.contrib.flask import Sentry
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import flask_jsglue as jsg
import logging

# JS Glue (allows use of Flask.url_for inside javascript)
jsglue = jsg.JSGlue()

# Feature Flags (allows turning on/off of features)
flags = FeatureFlag()

# Sentry error logging
sentry = Sentry(logging=True, level=logging.ERROR)

# Route Rate Limiter
limiter = Limiter(key_func=get_remote_address)
