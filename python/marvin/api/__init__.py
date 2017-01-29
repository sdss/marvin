
from flask import Blueprint, jsonify


def parse_params(request):
    """Parses the release from a POST Interaction request."""

    release = request.form['release'] if 'release' in request.form else None

    return release

apierrors = Blueprint('api_error_handlers', __name__)


@apierrors.errorhandler(422)
def handle_unprocessable_entity(err):
    # webargs attaches additional metadata to the `data` attribute
    data = getattr(err, 'data')
    if data:
        # Get validations from the ValidationError object
        messages = data['exc'].messages
    else:
        messages = ['Invalid request']
    print('422 error messages', messages)
    return jsonify({
        'messages': messages,
    }), 422


