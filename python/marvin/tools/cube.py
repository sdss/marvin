from flask_restful import Resource
from astropy.io import fits

#@api.resource('/cube/<mangaid>')
class Cube(Resource):
    def __init__(self,filename=None,mangaid=None,plateifu=None):
        assert filename is not None or mangaid is not None or plateifu is not None, 'Either filename, mangaid, or plateifu is required!'
        self.filename = filename
        self.mangaid = mangaid
        # Get by filename
        if self.filename:
            self.mode='file'
            self._openFile()
        else:
            self.hdu = None
        # Get by mangaid
        if self.mangaid:
            self.mode='db'
            self.getCubeFromMangaID()

    @classmethod
    def get(cls,mangaid=None):
        cube = cls(mangaid=mangaid)

    @classmethod
    def post(self):
        pass

    def getSpectrum(self, x, y):
        ''' currently: x,y array indices 
        ideally: x,y in arcsecond relative to cube center '''
        #spectrum = Spectrum(x,y)
        shape = self.flux.shape
        assert len(shape)==3, 'Dimensions of flux not = 3'
        assert x < shape[2] and y < shape[1], 'Input x,y coordinates greater than flux dimensions'
        return self.flux[:,y,x]

    def _openFile(self):
        self.hdu = fits.open(self.filename)

    @property
    def flux(self):
        if self.mode=='file':
            return self.hdu['FLUX'].data
        if self.mode=='db':
            return None

    def getCubeFromMangaID(self):
        pass

#@api.resource('cube/<mangaid>/spectrum/<specid>')
class Spectrum(Resource):
    def __init__(self, x=None,y=None):
        pass

    def get(self):
        pass

    def post(self):
        pass





