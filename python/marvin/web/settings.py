# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-28 15:32:49
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-17 13:15:31

from __future__ import print_function, division, absolute_import
import os
import datetime
import redis


class Config(object):
    FLASK_APP = os.environ.get('FLASK_APP', 'marvin.web.uwsgi_conf_files.app')
    SECRET_KEY = os.environ.get('MARVIN_SECRET', 'secret-key')
    FLASK_SECRET = SECRET_KEY
    APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
    APP_BASE = os.environ.get('MARVIN_BASE', 'marvin')
    projroot = os.path.abspath(os.path.join(APP_DIR, os.pardir, os.pardir, os.pardir))
    PROJECT_ROOT = os.environ.get('MARVIN_DIR', projroot)
    BCRYPT_LOG_ROUNDS = 13
    ASSETS_DEBUG = False
    DEBUG_TB_ENABLED = False  # Disable Debug toolbar
    DEBUG_TB_INTERCEPT_REDIRECTS = False
    MAIL_SERVER = ''
    MAIL_PORT = 587
    MAIL_USE_SSL = False
    MAIL_USERNAME = ''
    MAIL_PASSWORD = ''
    MAIL_DEFAULT_SENDER = ''
    GOOGLE_ANALYTICS = ''
    LOG_SQL_QUERIES = True
    FEATURE_FLAGS_NEW = True
    UPLOAD_FOLDER = os.environ.get("MARVIN_DATA_DIR", '/tmp/')
    LIB_PATH = os.path.join(APP_DIR, 'lib')
    ALLOWED_EXTENSIONS = set(['txt', 'csv'])
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    WTF_CSRF_ENABLED = True
    WTF_CSRF_SECRET_KEY = 'wtfsecretkey'
    USE_PROFILER = True  # Use the Flask Profiler Extension
    USE_SENTRY = False  # Turn off Sentry error logging
    FLASK_PROFILER = {
        "enabled": True,
        "storage": {
            "engine": "sqlite",
            "FILE": os.path.join(PROJECT_ROOT, 'flask_profiler.sql')
        },
        'endpointRoot': '{0}/profiler'.format(APP_BASE),
        "basicAuth": {
            "enabled": False
        },
        "ignore": [
            "/marvin/jsglue.js",
            "/marvin/static/.*",
            "/marvin/lib/.*",
            "/marvin/getgalidlist/"
        ]
    }
    # RATELIMIT_DEFAULT = '10/hour;100/day;2000 per year'
    RATELIMIT_STRATEGY = 'fixed-window-elastic-expiry'
    RATELIMIT_ENABLED = False

    # Flask-Session settings
    SESSION_TYPE = 'redis'
    SESSION_REDIS = redis.from_url(os.environ.get('SESSION_REDIS'))

    # Flask-Caching settings
    CACHE_TYPE = 'redis'  # Can be "memcached", "redis", etc.
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_REDIS_URL = os.environ.get("SESSION_REDIS")


class ProdConfig(Config):
    """Production configuration."""
    FLASK_ENV = os.environ.get('FLASK_ENV', 'production')
    ENV = 'prod'
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = 'postgresql://localhost/example'
    DEBUG_TB_ENABLED = False  # Disable Debug toolbar
    USE_X_SENDFILE = True
    USE_SENTRY = True
    SENTRY_DSN = os.environ.get('SENTRY_DSN', None)
    PERMANENT_SESSION_LIFETIME = datetime.timedelta(3600)
    JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(300)
    REMEMBER_COOKIE_DOMAIN = '.sdss.org'


class DevConfig(Config):
    """Development configuration."""
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    ENV = 'dev'
    DEBUG = True
    DB_NAME = 'dev.db'
    # Put the db file in project root
    DB_PATH = os.path.join(Config.PROJECT_ROOT, DB_NAME)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///{0}'.format(DB_PATH)
    DEBUG_TB_ENABLED = True
    ASSETS_DEBUG = True  # Don't bundle/minify static assets
    CACHE_TYPE = 'simple'  # Can be "memcached", "redis", etc.
    RATELIMIT_ENABLED = False


class TestConfig(Config):
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite://'
    BCRYPT_LOG_ROUNDS = 1  # For faster tests
    WTF_CSRF_ENABLED = False  # Allows form testing
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    USE_PROFILER = False  # Turn off the Flask Profiler extension
    RATELIMIT_ENABLED = False  # Turn off the Flask Rate Limiter
    #os.environ['PUBLIC_SERVER'] = 'True' # this breaks the debug server


class CustomConfig(object):
    ''' Project specific configuration.  Always gets appended to an above Config class '''
    os.environ['SAS_REDUX'] = 'sas/mangawork/manga/spectro/redux'
    os.environ['SAS_ANALYSIS'] = 'sas/mangawork/manga/spectro/analysis'
    os.environ['SAS_SANDBOX'] = 'sas/mangawork/manga/sandbox'
    release = os.environ.get('MARVIN_RELEASE', 'mangawork')
    os.environ['SAS_PREFIX'] = 'marvin' if release == 'mangawork' else 'dr15/marvin'
