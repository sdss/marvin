
from flask import Flask,Blueprint
from flask_restful import Api

def create_app(debug=False):


    from marvin.api.cube import CubeView
    from marvin.web.controllers.index import index_page

    app = Flask(__name__)
    api_bp = Blueprint('api', __name__)
    api = Api(api_bp)

    #api.add_resource(Cube, '/api/cubes/<string:mangaid>/',endpoint='mangaid')
    #api.add_resource(Cube.getSpectrum, '/api/cubes/<string:mangaid>/spectrum/',endpoint='spectrum')
    #app.register_blueprint(api_bp)
    #app.register_blueprint(index_page)
    CubeView.register(app)

    return app