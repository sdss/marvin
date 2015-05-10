#!/usr/bin/python

from __future__ import print_function

import os
import sys
import socket
from inspect import getmembers, isfunction
#from raven.contrib.flask import Sentry

import flask
import jinja_filters

def create_app(debug=False):
    app = flask.Flask(__name__, static_url_path='/manga/static')

    app.debug = debug
    print("{0}App '{1}' created.{2}".format('\033[92m', __name__, '\033[0m')) # to remove later

    # Define custom filters into the Jinja2 environment.
    # Any filters defined in the jinja_env submodule are made available.
    # See: http://stackoverflow.com/questions/12288454/how-to-import-custom-jinja2-filters-from-another-file-and-using-flask
    custom_filters = {name: function
                      for name, function in getmembers(jinja_filters)
                      if isfunction(function)}
    app.jinja_env.filters.update(custom_filters)

    if app.debug == False:
        # ----------------------------------------------------------
        # Set up getsentry.com logging - only use when in production
        # ----------------------------------------------------------        
        #dsn = 'https://f2cd2a5dbd7e45d2bd61faa74ae0b8c6:6fcddd5d410444f1a2de2f63441dec8f@app.getsentry.com/29141'
        #app.config['SENTRY_DSN'] = dsn
        #sentry = Sentry(app)

        # --------------------------------------
        # Configuration when running under uWSGI
        # --------------------------------------
        try:
            import uwsgi
            app.use_x_sendfile = True
        except ImportError:
            # not running under uWSGI (and presumably, nginx)
            pass

    # Change the implementation of "decimal" to a C-based version (much! faster)
    try:
        import cdecimal
        sys.modules["decimal"] = cdecimal
    except ImportError:
        pass # no available

    # Determine which configuration file should be loaded based on which
    # server we are running on. This value is set in the uWSGI config file for each server.
    
    # Add global SAS Redux path
    os.environ['SAS_REDUX'] = 'sas/mangawork/manga/spectro/redux'
    
    # Find which connection to make
    try: machine = os.environ['HOSTNAME']
    except: machine = None
    
    try: localhost = bool(os.environ['MANGA_LOCALHOST'])
    except: localhost = machine == 'manga'
    
    try: utah = os.environ['UUFSCELL']='kingspeak.peaks'
    except: utah = None

    if app.debug: 
        if localhost: server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'configuration_files','localhost.cfg')
        elif utah: server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'configuration_files','utah.cfg')
        else:
            print("Trying to run in debug mode, but not running\n"
                  "on a development machine that has database access.")
            sys.exit(1)
    else:
        try:
            import uwsgi
            server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'configuration_files', uwsgi.opt['flask-config-file'])
        except ImportError:
            print("Trying to run in production mode, but not running under uWSGI.\n"
                       "You might try running again with the '--debug' flag.")
            sys.exit(1)

    print("Loading config file: {0}".format(server_config_file))
    app.config.from_pyfile(server_config_file)

    print("Server_name = {0}".format(app.config["SERVER_NAME"]))

    # This "with" is necessary to prevent exceptions of the form:
    #    RuntimeError: working outside of application context
    with app.app_context():
        from .model.database import db
    
    # -------------------
    # Register blueprints
    # -------------------
    from .controllers.index import index_page
    from .controllers.search import search_page
    from .controllers.current import current_page
    from .controllers.plate import plate_page
    from .controllers.images import image_page
    from .controllers.comments import comment_page
    from .controllers.feedback import feedback_page
    
    url_prefix = '' if localhost else '/manga'

    app.register_blueprint(index_page, url_prefix=url_prefix)
    app.register_blueprint(search_page, url_prefix=url_prefix)
    app.register_blueprint(current_page, url_prefix=url_prefix)
    app.register_blueprint(plate_page, url_prefix=url_prefix)
    app.register_blueprint(image_page, url_prefix=url_prefix)
    app.register_blueprint(comment_page, url_prefix=url_prefix)
    app.register_blueprint(feedback_page, url_prefix=url_prefix)
    
    return app

# Perform early app setup here.






