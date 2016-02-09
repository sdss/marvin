from flask.ext.classy import FlaskView, route
from marvin.tools.cube import Cube

class CubeView(FlaskView):
    route_base='/cubes/'

    def index(self):
        return 'cube'

    def get(self,mangaid):
        cube = Cube(mangaid=mangaid)
        return 'cube-mangaid: {0},{1},{2}'.format(mangaid,cube.ra,cube.dec)

    @route('/<mangaid>/spectra/x=<x>/y=<y>/')
    def getSpectra(self,mangaid=None,x=None,y=None):
        cube = Cube(mangaid=mangaid)
        spectrum = cube.getSpectrum(15,15)
        return {'spectrum': 'spectrum: {0},{1} for mangaid {2},{3}'
                .format(x, y, mangaid, spectrum)}
