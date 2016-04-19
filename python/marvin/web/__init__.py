#!/usr/bin/python

from __future__ import print_function
from flask import Flask, Blueprint
from flask_restful import Api
from flask_jsglue import JSGlue
from inspect import getmembers, isfunction
from marvin.utils.general.general import getDbMachine
from marvin import config
from flask_featureflags import FeatureFlag
from raven.contrib.flask import Sentry
import marvin.web.jinja_filters
import sys
import os


def create_app(debug=False):

    from marvin.api.cube import CubeView
    from marvin.api.rss import RSSView
    from marvin.api.spaxel import SpaxelView
    from marvin.api.query import QueryView
    from marvin.api.general import GeneralRequestsView
    from marvin.web.controllers.index import index
    from marvin.web.controllers.galaxy import galaxy

    # ----------------------------------
    # Create App
    app = Flask(__name__, static_url_path='/marvin/static')
    api = Blueprint("api", __name__, url_prefix='/api')
    app.debug = debug
    jsglue = JSGlue(app)

    # Define custom filters into the Jinja2 environment.
    # Any filters defined in the jinja_env submodule are made available.
    # See: http://stackoverflow.com/questions/12288454/how-to-import-custom-jinja2-filters-from-another-file-and-using-flask
    custom_filters = {name: function for name, function in getmembers(jinja_filters) if isfunction(function)}
    app.jinja_env.filters.update(custom_filters)

    # ----------------------------------
    # Initialize logging + Sentry + UWSGI config for Production Marvin
    if app.debug is False:
        ''' set up logging here, or try to use the tools logger '''

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
    os.environ['SAS_PREFIX'] = 'marvin' if release == 'mangawork' else 'dr13/marvin'
    url_prefix = '/marvin' if local else '/{0}'.format(os.environ['SAS_PREFIX'])

    # ----------------------------------
    # Load the appropriate Flask configuration file for debug or production
    if app.debug:
        if local:
            server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration', 'localhost.cfg')
        elif connection == 'utah':
            server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration', 'utah.cfg')
        else:
            print("Trying to run in debug mode, but not running on a development machine that has database access.")
            sys.exit(1)
    else:
        try:
            import uwsgi
            server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration', uwsgi.opt['flask-config-file'])
        except ImportError:
            print("Trying to run in production mode, but not running under uWSGI. You might try running again with the '--debug' flag.")
            sys.exit(1)

    print('Loading config file: {0}'.format(server_config_file))
    app.config.from_pyfile(server_config_file)

    # ----------------------------------
    # Initialize feature flags
    feature_flags = FeatureFlag(app)
    # configFeatures(debug=app.debug)

    # Update any config parameters
    app.config["UPLOAD_FOLDER"] = os.environ.get("MARVIN_DATA_DIR", None)

    # ----------------------------------
    # Registration
    #
    # API route registration
    CubeView.register(api)
    RSSView.register(api)
    SpaxelView.register(api)
    GeneralRequestsView.register(api)
    QueryView.register(api)
    app.register_blueprint(api)

    # Web route registration
    app.register_blueprint(index, url_prefix=url_prefix)
    app.register_blueprint(galaxy, url_prefix=url_prefix)

    return app
