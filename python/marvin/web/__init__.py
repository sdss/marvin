#!/usr/bin/python

from __future__ import print_function
from flask import Flask, Blueprint
from flask_restful import Api
from inspect import getmembers, isfunction
import sys
import os
from marvin.utils.general.general import getDbMachine
from flask_featureflags import FeatureFlag
import jinja_filters


def create_app(debug=False):

    from marvin.api.cube import CubeView
    from marvin.api.general import GeneralRequestsView
    from marvin.web.controllers.index import index

    # ----------------------------------
    # Create App
    app = Flask(__name__, static_url_path='/marvin/static')
    api = Blueprint("api", __name__, url_prefix='/api')
    app.debug = debug

    # Define custom filters into the Jinja2 environment.
    # Any filters defined in the jinja_env submodule are made available.
    # See: http://stackoverflow.com/questions/12288454/how-to-import-custom-jinja2-filters-from-another-file-and-using-flask
    custom_filters = {name: function for name, function in getmembers(jinja_filters) if isfunction(function)}
    app.jinja_env.filters.update(custom_filters)

    # ----------------------------------
    # Initialize logging + Sentry + UWSGI config for Production Marvin
    if app.debug is False:
        pass

    # Change the implementation of "decimal" to a C-based version (much! faster)
    try:
        import cdecimal
        sys.modules["decimal"] = cdecimal
    except ImportError:
        pass  # no available

    # Find which connection to make
    connection = getDbMachine()
    local = connection == 'local'

    # ----------------------------------
    # Set some environment variables
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
    GeneralRequestsView.register(api)
    app.register_blueprint(api)

    # Web route registration
    app.register_blueprint(index, url_prefix=url_prefix)

    return app
