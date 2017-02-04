
# from flask import Blueprint, jsonify, make_response
# from marvin.utils.db import get_traceback
# import sys

#
# This should all be moved into the Brain and abstracted since it is general useful standardized stuff (what?)


def parse_params(request):
    """Parses the release from a POST Interaction request."""

    release = request.form['release'] if 'release' in request.form else None

    return release


# theapi = Blueprint("api", __name__, url_prefix='/marvin2/api')


# @theapi.errorhandler(422)
# def handle_unprocessable_entity(err):
#     # webargs attaches additional metadata to the `data` attribute
#     data = getattr(err, 'data')
#     if data:
#         # Get validations from the ValidationError object
#         messages = data['messages']
#     else:
#         messages = ['Invalid request']
#     return jsonify({
#         'validation_errors': messages,
#     }), 422


# @theapi.errorhandler(500)
# def internal_server_error(err):
#     messages = {'error': 'internal_server_error',
#                 'message': err.description,
#                 'traceback': get_traceback(asstring=True)}
#     return jsonify({
#         'api_error': messages,
#     }), 500
