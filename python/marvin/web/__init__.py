
from flask import Flask,Blueprint
from flask_restful import Api

def create_app(debug=False):


    from marvin.tools.cube import Cube,Spectrum
    from marvin.web.controllers.index import index_page

    app = Flask(__name__)
    api_bp = Blueprint('api', __name__)
    api = Api(api_bp)

    api.add_resource(Cube, '/api/cubes/<string:mangaid>/',endpoint='mangaid')
    app.register_blueprint(api_bp)
    app.register_blueprint(index_page)

    return app