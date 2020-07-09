# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-01-27 14:26:40
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-12 14:14:41

from __future__ import print_function, division, absolute_import
from flask import request, current_app as app
from flask import Blueprint, jsonify, render_template, g
from marvin.utils.db import get_traceback
from marvin.web.extensions import sentry


errors = Blueprint('error_handlers', __name__)


def make_error_json(error, name, code, data=None):
    ''' creates the error json dictionary for API errors '''
    shortname = name.lower().replace(' ', '_')
    messages = {'error': shortname,
                'message': error.description if hasattr(error, 'description') else None,
                'status_code': code,
                'traceback': get_traceback(asstring=True)}
    if data:
        return jsonify({'validation_errors': data}), code
    else:
        return jsonify({'api_error': messages}), code


def make_error_page(app, name, code, sentry=None, data=None, exception=None):
    ''' creates the error page dictionary for web errors '''
    shortname = name.lower().replace(' ', '_')
    error = {}
    error['title'] = 'Marvin | {0}'.format(name)
    error['page'] = request.url
    error['event_id'] = g.get('sentry_event_id', None)
    error['data'] = data
    error['name'] = name
    error['code'] = code
    error['message'] = exception.description if exception and hasattr(exception, 'description') else None
    if app.config['USE_SENTRY'] and sentry:
        error['public_dsn'] = sentry.client.get_public_dsn('https')
    app.logger.error('{0} Exception {1}'.format(name, error))
    return render_template('errors/{0}.html'.format(shortname), **error), code

# ----------------
# Error Handling
# ----------------


def _is_api(request):
    ''' Checks if the error comes from the api '''
    return request.blueprint == 'api' or 'api' in request.url


@errors.app_errorhandler(404)
def page_not_found(error):
    name = 'Page Not Found'
    if _is_api(request):
        return make_error_json(error, name, 404)
    else:
        return make_error_page(app, name, 404, sentry=sentry, exception=error)


@errors.app_errorhandler(500)
def internal_server_error(error):
    name = 'Internal Server Error'
    if _is_api(request):
        return make_error_json(error, name, 500)
    else:
        return make_error_page(app, name, 500, sentry=sentry, exception=error)


@errors.app_errorhandler(400)
def bad_request(error):
    name = 'Bad Request'
    if _is_api(request):
        return make_error_json(error, name, 400)
    else:
        return make_error_page(app, name, 400, sentry=sentry, exception=error)


@errors.app_errorhandler(405)
def method_not_allowed(error):
    name = 'Method Not Allowed'
    if _is_api(request):
        return make_error_json(error, name, 405)
    else:
        return make_error_page(app, name, 405, sentry=sentry, exception=error)


@errors.app_errorhandler(422)
def handle_unprocessable_entity(error):
    name = 'Unprocessable Entity'
    data = getattr(error, 'data')
    if data:
        # Get validations from the ValidationError object
        messages = data['messages']
    else:
        messages = ['Invalid request']

    if _is_api(request):
        return make_error_json(error, name, 422, data=messages)
    else:
        return make_error_page(app, name, 422, sentry=sentry, data=messages, exception=error)


@errors.app_errorhandler(429)
def rate_limit_exceeded(error):
    name = 'Rate Limit Exceeded'
    if _is_api(request):
        return make_error_json(error, name, 429)
    else:
        return make_error_page(app, name, 429, sentry=sentry, exception=error)


@errors.app_errorhandler(504)
def gateway_timeout(error):
    name = 'Gateway Timeout'
    if _is_api(request):
        return make_error_json(error, name, 504)
    else:
        return make_error_page(app, name, 504, sentry=sentry, exception=error)


@errors.app_errorhandler(413)
def entity_too_large(error):
    name = 'Request Entity Too Large'
    if _is_api(request):
        return make_error_json(error, name, 413)
    else:
        return make_error_page(app, name, 413, sentry=sentry, exception=error)
