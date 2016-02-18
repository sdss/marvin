
from flask import Flask, Blueprint
from flask_restful import Api


def create_app(debug=False):

    from marvin.api.cube import CubeView
    from marvin.api.general import GeneralRequestsView
    from marvin.web.controllers.index import index

    app = Flask(__name__)
    api = Blueprint("api", __name__)

    # API route registration
    CubeView.register(api)
    GeneralRequestsView.register(api)
    app.register_blueprint(api)

    # Web route registration
    app.register_blueprint(index)

    return app
