# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Filename: celery_worker.py
# Project: web
# Author: Brian Cherinka
# Created: Friday, 16th August 2019 5:30:18 pm
# License: BSD 3-clause "New" or "Revised" License
# Copyright (c) 2019 Brian Cherinka
# Last Modified: Saturday, 7th September 2019 4:51:12 pm
# Modified By: Brian Cherinka

#!/usr/bin/env python

# Celery needs a worker server to receive and manage its tasks.  See
# https://docs.celeryproject.org/en/latest/userguide/workers.html
#
# This script sets up the celery worker so it has knowledge of the Flask application context
# See the blog post:
# https://blog.miguelgrinberg.com/post/celery-and-the-flask-application-factory-pattern
#
# which is continuing on from this blog post:
# https://blog.miguelgrinberg.com/post/using-celery-with-flask
#
# To run the celery worker from the command line, type
# >>> celery worker -A celery_worker.celery --loglevel info
#
# To run the celery worker as a daemon, see
# https://docs.celeryproject.org/en/latest/userguide/daemonizing.html#daemonizing
#

from __future__ import print_function, division, absolute_import
from marvin.web import celery, create_app

app = create_app()
app.app_context().push()
