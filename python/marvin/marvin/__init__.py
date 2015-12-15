#!/usr/bin/python

from __future__ import print_function

import os
import sys
import socket
from os.path import join
from inspect import getmembers, isfunction
from raven.contrib.flask import Sentry
from flask_featureflags import FeatureFlag
from flask_restful import Api

import flask
import jinja_filters

def create_app(debug=False):
    app = flask.Flask(__name__, static_url_path='/marvin/static')
    api_bp = flask.Blueprint('api', __name__)
    api = Api(api_bp)

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
        # Set up logging on sas to logs dir
        # ----------------------------------------------------------
        import logging
        from logging.handlers import RotatingFileHandler
        try: logfile = join(os.environ['SAS_LOGS_DIR'],'mylogger.log')
        except: logfile = None
        if logfile:
            file_handler = RotatingFileHandler(logfile, maxBytes=1024 * 1024 * 100, backupCount=20)
            file_handler.setLevel(logging.NOTSET)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            app.logger.addHandler(file_handler)
            app.logger.warning("HELLO")
        app.logger.info("WORLD")
        app.logger.error("!")


        # ----------------------------------------------------------
        # Set up getsentry.com logging - only use when in production
        # ----------------------------------------------------------        
        dsn = 'https://989c330efbc346c7916e97b4edbf6b80:ae563b713f744429a8fd5ce55727b66d@app.getsentry.com/52254'
        app.config['SENTRY_DSN'] = dsn
        sentry = Sentry(app,logging=True,level=logging.ERROR)
        

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
    os.environ['SAS_ANALYSIS'] = 'sas/mangawork/manga/spectro/analysis'
    os.environ['SAS_SANDBOX'] = 'sas/mangawork/manga/sandbox'

    # Find which connection to make
    try: machine = os.environ['HOSTNAME']
    except: machine = None
    
    try: localhost = bool(os.environ['MANGA_LOCALHOST'])
    except: localhost = machine == 'manga'
    
    try: utah = os.environ['UUFSCELL'] == 'kingspeak.peaks'
    except: utah = None

    try: sasvm = 'sas-vm' in os.environ['HOSTNAME']
    except: sasvm = None

    if app.debug: 
        if localhost: server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'configuration_files','localhost.cfg')
        elif utah or sasvm: server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'configuration_files','utah.cfg')
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

    # ----------------
    # Error Handling
    # ----------------
    @app.errorhandler(404)
    def page_not_found(e):
        error={}
        error['title']='Marvin | Page Not Found'
        error['page'] = flask.request.url
        flask.session['vermode']='MPL'
        app.logger.error('Page Not Found Exception {0}'.format(e))
        return flask.render_template('errors/page_not_found.html',**error),404 

    @app.errorhandler(500)
    def internal_server_error(e):
        error={}
        error['title']='Marvin | Internal Server Error'
        flask.session['vermode']='MPL'
        app.logger.error('Internal Server Error Exception {0}'.format(e))        
        return flask.render_template('errors/internal_server_error.html',**error),500

    @app.errorhandler(400)
    def bad_request(e):
        error={}
        error['title']='Marvin | Bad Request'
        flask.session['vermode']='MPL'
        app.logger.error('Bad Request Exception {0}'.format(e))        
        return flask.render_template('errors/bad_request.html',**error),400

    @app.errorhandler(405)
    def method_not_allowed(e):
        error={}
        error['title']='Marvin | Method Not Allowed'
        flask.session['vermode']='MPL'
        app.logger.error('Method Not Allowed Exception {0}'.format(e))        
        return flask.render_template('errors/method_not_allowed.html',**error),405

    # -------------
    # Initialize feature flags
    feature_flags = FeatureFlag(app)
    #configFeatures(debug=app.debug)

    # Update any config parameters
    app.config["UPLOAD_FOLDER"] = os.getenv("MARVIN_DATA_DIR")

    # -------------

    # -------------------
    # Register blueprints
    # -------------------
    from .controllers.index import index_page
    from .controllers.search import search_page
    from .controllers.current import current_page
    from .controllers.plate import plate_page, TestPlate, MangaID, MangaIDList
    from .controllers.images import image_page
    from .controllers.comments import comment_page
    from .controllers.feedback import feedback_page
    from .controllers.explore import explore_page
    from .controllers.documentation import doc_page
    from .controllers.tests import test_page
    
    try: release = os.environ['MARVIN_RELEASE'] 
    except: release = 'mangawork'

    os.environ['SAS_PREFIX'] = 'marvin' if release == 'mangawork' else 'dr13/marvin'
    url_prefix = '' if localhost else '/{0}'.format(os.environ['SAS_PREFIX']) 

    # register API hooks
    api.add_resource(TestPlate, '/testplates/<string:plateid>', endpoint='testplate')
    api.add_resource(MangaIDList, '/api/mangaids/', endpoint='mangaids')
    api.add_resource(MangaID, '/api/mangaids/<string:mangaid>/', endpoint='mangaid')

    # Register Flask App pages
    app.register_blueprint(api_bp)
    app.register_blueprint(index_page, url_prefix=url_prefix)
    app.register_blueprint(search_page, url_prefix=url_prefix)
    app.register_blueprint(current_page, url_prefix=url_prefix)
    app.register_blueprint(plate_page, url_prefix=url_prefix)
    app.register_blueprint(image_page, url_prefix=url_prefix)
    app.register_blueprint(comment_page, url_prefix=url_prefix)
    app.register_blueprint(feedback_page, url_prefix=url_prefix)
    app.register_blueprint(explore_page, url_prefix=url_prefix)
    app.register_blueprint(doc_page, url_prefix=url_prefix)
    app.register_blueprint(test_page, url_prefix=url_prefix)
    
    return app

# Perform early app setup here.






