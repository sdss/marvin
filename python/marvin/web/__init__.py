
from flask import Flask
from flask_restful import Api

def create_app(debug=False):


    from marvin.tools.cube import Cube,Spectrum
    from marvin.web.controllers.index import index_page

    app = Flask(__name__)
    api = Api(app)

    api.add_resource(Cube, '/api/cubes/<string:mangaid>/',endpoint='mangaid')
    app.register_blueprint(index_page)

    return app