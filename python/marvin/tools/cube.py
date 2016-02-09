
import requests

class Cube(object):

    def __init__(self,filename=None,mangaid=None,plateifu=None):
        assert filename is not None or mangaid is not None or plateifu is not None, 'Either filename, mangaid, or plateifu is required!'
        print('init',mangaid)
        self.mode = None
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
            try:
                self.mode='db'
                self._getCubeFromMangaID()
            except:
                self.mode = None

    def getSpectrum(self, x, y):
        ''' currently: x,y array indices
        ideally: x,y in arcsecond relative to cube center '''
        #spectrum = Spectrum(x,y)
        if self.mode == 'file':
            ''' local (client has a file) '''
            shape = self.flux.shape
            assert len(shape)==3, 'Dimensions of flux not = 3'
            assert x < shape[2] and y < shape[1], 'Input x,y coordinates greater than flux dimensions'
            return self.flux[:,y,x]
        elif self.mode == 'db':
            ''' db means local (client) has db '''
            return self._cube.spaxels[0].flux
        else:
            ''' local (client) has nothing '''
            response = requests.get(
                'http://5aafb8e.ngrok.com/cubes/{0}/spectra/x={1}/y={2}/'
                .format(self.mangaid, x, y))
            return response.text

    def _openFile(self):
        self.hdu = fits.open(self.filename)

    @property
    def flux(self):
        if self.mode=='file':
            return self.hdu['FLUX'].data
        if self.mode=='db':
            return None

    def _getCubeFromMangaID(self):
        ''' server-side code '''
        from ..db.database import db
        import sdss.internal.database.utah.mangadb.DataModelClasses as datadb
        session = db.Session()
        self._cube = session.query(datadb.Cube).join(datadb.PipelineInfo,datadb.PipelineVersion).filter(datadb.PipelineVersion.version=='trunk',datadb.Cube.mangaid==self.mangaid).first()
        print('cube',self._cube, len(self._cube.spaxels))
        self.ifu = self._cube.ifu.name
        self.ra = self._cube.ra
        self.dec = self._cube.dec
        self.plate = self._cube.plate
