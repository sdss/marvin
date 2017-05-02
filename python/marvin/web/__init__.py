#!/usr/bin/python

from __future__ import print_function, division
from flask import Flask, Blueprint, send_from_directory
from flask_restful import Api
from flask import session, request, render_template, g, jsonify
import flask_jsglue as jsg
import flask_profiler
from inspect import getmembers, isfunction
from raven.contrib.flask import Sentry
from brain.utils.general.general import getDbMachine
from marvin import config, log
from flask_featureflags import FeatureFlag
from marvin.web.jinja_filters import jinjablue
from marvin.web.error_handlers import weberrors as web
from marvin.utils.db import get_traceback
from marvin.web.web_utils import updateGlobalSession, make_error_page, send_request, make_error_json
import sys
import os
import logging


# ================================================================================

def register_blueprints(app=None):
    '''
    Register the code associated with each URL paths. Manually add each new
    controller file you create here.
    '''
    from marvin.web.controllers.index import index

    app.register_blueprint(index)

# ================================================================================


def create_app(debug=False, local=False, use_profiler=True):

    #from marvin.api import theapi as api
    from marvin.api.cube import CubeView
    from marvin.api.maps import MapsView
    from marvin.api.modelcube import ModelCubeView
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
    from marvin.web.controllers.users import users

    # ----------------------------------
    # Create App
    marvin_base = os.environ.get('MARVIN_BASE', 'marvin2')
    app = Flask(__name__, static_url_path='/{0}/static'.format(marvin_base))
    api = Blueprint("api", __name__, url_prefix='/{0}/api'.format(marvin_base))
    app.debug = debug
    jsg.JSGLUE_JS_PATH = '/{0}/jsglue.js'.format(marvin_base)
    jsglue = jsg.JSGlue(app)

    # Add Marvin Logger
    app.logger.addHandler(log)

    # Setup the profile configuration
    app.config["flask_profiler"] = {
        "enabled": True,
        "storage": {
            "engine": "sqlite",
            "FILE": os.path.join(os.environ['MARVIN_DIR'], 'flask_profiler.sql')
        },
        'endpointRoot': '{0}/profiler'.format(marvin_base),
        "basicAuth": {
            "enabled": False
        }
    }

    # ----------------------------------
    # Initialize logging + Sentry + UWSGI config for Production Marvin
    if app.debug is False:

        # ----------------------------------------------------------
        # Set up getsentry.com logging - only use when in production
        dsn = os.environ.get('SENTRY_DSN', None)
        app.config['SENTRY_DSN'] = dsn
        sentry = Sentry(app, logging=True, level=logging.ERROR)

        # --------------------------------------
        # Configuration when running under uWSGI
        try:
            import uwsgi
            app.use_x_sendfile = True
        except ImportError:
            pass
    else:
        sentry = None
        config.use_sentry = False
        config.add_github_message = False

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
    local = (connection == 'local') or local

    # ----------------------------------
    # Set some environment variables
    config._inapp = True
    os.environ['SAS_REDUX'] = 'sas/mangawork/manga/spectro/redux'
    os.environ['SAS_ANALYSIS'] = 'sas/mangawork/manga/spectro/analysis'
    os.environ['SAS_SANDBOX'] = 'sas/mangawork/manga/sandbox'
    release = os.environ.get('MARVIN_RELEASE', 'mangawork')
    os.environ['SAS_PREFIX'] = 'marvin2' if release == 'mangawork' else 'dr13/marvin'
    url_prefix = '/marvin2' if local else '/{0}'.format(marvin_base)

    # ----------------------------------
    # Load the appropriate Flask configuration file for debug or production
    if app.debug:
        if local:
            server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration', 'localhost.cfg')
        elif connection == 'utah':
            server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration', 'utah.cfg')
        else:
            server_config_file = None
            app.logger.debug("Trying to run in debug mode, but not running on a development machine that has database access.")
            # sys.exit(1)
    else:
        try:
            import uwsgi
            server_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration', uwsgi.opt['flask-config-file'])
        except ImportError:
            app.logger.debug("Trying to run in production mode, but not running under uWSGI. You might try running again with the '--debug' flag.")
            sys.exit(1)

    if server_config_file:
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
    @app.route('/{0}/lib/<path:filename>'.format(marvin_base))
    def lib(filename):
        return send_from_directory(app.config["LIB_PATH"], filename)

    # Register update global session
    @app.before_request
    def global_update():
        ''' updates the global session / config '''
        updateGlobalSession()
        # send_request()

    # ----------------
    # Error Handling
    # ----------------

    def _is_api(request):
        ''' Checks if the error comes from the api '''
        return request.blueprint == 'api' or 'api' in request.url

    @app.errorhandler(404)
    def page_not_found(error):
        name = 'Page Not Found'
        if _is_api(request):
            return make_error_json(error, name, 404)
        else:
            return make_error_page(app, name, 404, sentry=sentry)

    @app.errorhandler(500)
    def internal_server_error(error):
        name = 'Internal Server Error'
        if _is_api(request):
            return make_error_json(error, name, 500)
        else:
            return make_error_page(app, name, 500, sentry=sentry)

    @app.errorhandler(400)
    def bad_request(error):
        name = 'Bad Request'
        if _is_api(request):
            return make_error_json(error, name, 400)
        else:
            return make_error_page(app, name, 400, sentry=sentry)

    @app.errorhandler(405)
    def method_not_allowed(error):
        name = 'Method Not Allowed'
        if _is_api(request):
            return make_error_json(error, name, 405)
        else:
            return make_error_page(app, name, 405, sentry=sentry)

    @app.errorhandler(422)
    def handle_unprocessable_entity(error):
        name = 'Unprocessable Entity'
        data = getattr(error, 'data')
        if data:
            # Get validations from the ValidationError object
            messages = data['messages']
        else:
            messages = ['Invalid request']

        if _is_api(request):
            return make_error_json(error, name, 422)
        else:
            return make_error_page(app, name, 422, sentry=sentry, data=messages)

    # These should be moved into the api module but they need the api blueprint to be registered on
    # I had these in the api.__init__ with the api blueprint defined there as theapi
    # which was imported here, see line 39
    # This should all be moved into the Brain and abstracted since it is
    # general useful standardized stuff (what?)

    @api.errorhandler(422)
    def handle_unprocessable_entity(err):
        # webargs attaches additional metadata to the `data` attribute
        data = getattr(err, 'data')
        if data:
            # Get validations from the ValidationError object
            messages = data['messages']
        else:
            messages = ['Invalid request']
        return jsonify({
            'validation_errors': messages,
        }), 422

    # ----------------------------------
    # Registration
    config.use_sentry = False
    #
    # API route registration
    CubeView.register(api)
    MapsView.register(api)
    ModelCubeView.register(api)
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
    app.register_blueprint(users, url_prefix=url_prefix)

    # Register all custom Jinja filters in the file.
    app.register_blueprint(jinjablue)

    # Register error handlers
    app.register_blueprint(web)

    # Initialize the Flask-Profiler ; see results at localhost:portnumber/flask-profiler
    if use_profiler:
        try:
            flask_profiler.init_app(app)
        except Exception as e:
            pass

    return app
