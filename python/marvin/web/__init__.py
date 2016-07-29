#!/usr/bin/python

from __future__ import print_function, division
from flask import Flask, Blueprint, send_from_directory
from flask_restful import Api
import flask_jsglue as jsg
from flask_cors import CORS
import flask_profiler
from inspect import getmembers, isfunction
from brain.utils.general.general import getDbMachine
from marvin import config, log
from flask_featureflags import FeatureFlag
from raven.contrib.flask import Sentry
from marvin.web.jinja_filters import jinjablue
import sys
import os


# ================================================================================

def register_blueprints(app=None):
    '''
    Register the code associated with each URL paths. Manually add each new
    controller file you create here.
    '''
    from .controllers.index import index_page

    app.register_blueprint(index_page)

# ================================================================================


def create_app(debug=False):

    from marvin.api.cube import CubeView
    from marvin.api.maps import MapsView
    from marvin.api.plate import PlateView
    from marvin.api.rss import RSSView
    from marvin.api.spaxel import SpaxelView
    from marvin.api.query import QueryView
    from marvin.api.general import GeneralRequestsView
    from marvin.web.controllers.index import index
    from marvin.web.controllers.galaxy import galaxy
    from marvin.web.controllers.search import search
    from marvin.web.controllers.plate import plate
    from marvin.web.controllers.images import images

    # ----------------------------------
    # Create App
    app = Flask(__name__, static_url_path='/marvin2/static')
    api = Blueprint("api", __name__, url_prefix='/marvin2/api')
    app.debug = debug
    jsg.JSGLUE_JS_PATH = '/marvin2/jsglue.js'
    jsglue = jsg.JSGlue(app)
    CORS(app, origins='*')

    # Logger
    app.logger.addHandler(log)

    # Setup the profile configuration
    app.config["flask_profiler"] = {
        "enabled": True,
        "storage": {
            "engine": "sqlite"
        },
        "basicAuth": {
            "enabled": True,
            "username": "admin",
            "password": "admin"
        }
    }

    # ----------------------------------
    # Initialize logging + Sentry + UWSGI config for Production Marvin
    if app.debug is False:

        # ----------------------------------------------------------
        # Set up getsentry.com logging - only use when in production
        # dsn = 'https://989c330efbc346c7916e97b4edbf6b80:ae563b713f744429a8fd5ce55727b66d@app.getsentry.com/52254'
        # app.config['SENTRY_DSN'] = dsn
        # sentry = Sentry(app, logging=True, level=logging.ERROR)

        # --------------------------------------
        # Configuration when running under uWSGI
        try:
            import uwsgi
            app.use_x_sendfile = True
        except ImportError:
            pass

    # Change the implementation of "decimal" to a C-based version (much! faster)
    #
    # This is producing this error [ Illegal value: Decimal('3621.59598486') ]on some of the remote API tests with getSpectrum
    # Turning this off for now
    #
    # try:
    #    import cdecimal
    #    sys.modules["decimal"] = cdecimal
    # except ImportError:
    #    pass  # no available

    # Find which connection to make
    connection = getDbMachine()
    local = connection == 'local'

    # ----------------------------------
    # Set some environment variables
    config._inapp = True
    os.environ['SAS_REDUX'] = 'sas/mangawork/manga/spectro/redux'
    os.environ['SAS_ANALYSIS'] = 'sas/mangawork/manga/spectro/analysis'
    os.environ['SAS_SANDBOX'] = 'sas/mangawork/manga/sandbox'
    release = os.environ.get('MARVIN_RELEASE', 'mangawork')
    os.environ['SAS_PREFIX'] = 'marvin2' if release == 'mangawork' else 'dr13/marvin'
    url_prefix = '/marvin2' if local else '/{0}'.format(os.environ['SAS_PREFIX'])

    # ----------------------------------
    # Load the appropriate Flask configuration file for debug or production
    if app.debug:
        if local:
            server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration', 'localhost.cfg')
        elif connection == 'utah':
            server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration', 'utah.cfg')
        else:
            app.logger.debug("Trying to run in debug mode, but not running on a development machine that has database access.")
            sys.exit(1)
    else:
        try:
            import uwsgi
            server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration', uwsgi.opt['flask-config-file'])
        except ImportError:
            app.logger.debug("Trying to run in production mode, but not running under uWSGI. You might try running again with the '--debug' flag.")
            sys.exit(1)

    app.logger.info('Loading config file: {0}'.format(server_config_file))
    app.config.from_pyfile(server_config_file)

    # ----------------------------------
    # Initialize feature flags
    feature_flags = FeatureFlag(app)
    # configFeatures(debug=app.debug)

    # Update any config parameters
    app.config["UPLOAD_FOLDER"] = os.environ.get("MARVIN_DATA_DIR", None)
    app.config["LIB_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')

    # Add lib directory as a new static path
    @app.route('/marvin2/lib/<path:filename>')
    def lib(filename):
        return send_from_directory(app.config["LIB_PATH"], filename)

    # ----------------------------------
    # Registration
    #
    # API route registration
    CubeView.register(api)
    MapsView.register(api)
    PlateView.register(api)
    RSSView.register(api)
    SpaxelView.register(api)
    GeneralRequestsView.register(api)
    QueryView.register(api)
    app.register_blueprint(api)

    # Web route registration
    app.register_blueprint(index, url_prefix=url_prefix)
    app.register_blueprint(galaxy, url_prefix=url_prefix)
    app.register_blueprint(search, url_prefix=url_prefix)
    app.register_blueprint(plate, url_prefix=url_prefix)
    app.register_blueprint(images, url_prefix=url_prefix)

    # Register all custom Jinja filters in the file.
    app.register_blueprint(jinjablue)

    # Initialize the Flask-Profiler ; see results at localhost:portnumber/flask-profiler
    flask_profiler.init_app(app)

    return app
