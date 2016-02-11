from __future__ import print_function
import numpy as np
from astropy.io import fits
from marvin import config
from marvin.api.api import Interaction


class Cube(object):

    def __init__(self, filename=None, plateifu=None, mangaid=None):
        args = [filename, plateifu, mangaid]
        errmsg = 'Enter filename, plateifu, or mangaid!'
        assert any(args), errmsg
        assert sum([bool(arg) for arg in args]) == 1, errmsg

        self.filename = filename
        self.mangaid = mangaid
        self.plateifu = plateifu

        # convert from mangaid to plateifu
        # print warning if multiple plateifus correspond to one mangaid
        # FIX
        mangaid_to_plateifu = {'1-209232': '8485-1901'}
        if self.mangaid:
            self.plateifu = mangaid_to_plateifu[self.mangaid]

        if config.mode == 'local':
            if self.filename:
                self._openFile()
            else:
                try:
                    # self._getCubeFromPlateIFU()
                    self._getCubeFromMangaID()
                except RuntimeError:
                    # convert from plateifu to filename (sdss_access)
                    self.filename = plateifu_to_filename(self.plateifu)
                    try:
                        self._openFile()
                    except IOError as e:
                        if config.download:
                            # download file via sdsssync
                            # self._openFile()
                            pass
                        else:
                            raise Exception('File does not exist locally') from e
        elif config.mode == 'remote':
            if self.filename:
                raise FileNotFoundError('Cannot open file remotely.')
            else:
                # use API
                pass

    def getSpectrum(self, x, y):
        ''' currently: x,y array indices
        ideally: x,y in arcsecond relative to cube center '''
        # spectrum = Spectrum(x,y)
        if config.mode == 'file':
            ''' local (client has a file) '''
            shape = self.flux.shape
            assert len(shape) == 3, 'Dimensions of flux not = 3'
            assert x < shape[2] and y < shape[1], 'Input x,y coordinates greater than flux dimensions'
            return self.flux[:, y, x]
        elif config.mode == 'db':
            ''' db means local (client) has db '''
            return self._cube.spaxels[0].flux
        else:
            ''' local (client) has nothing '''
            route = 'cubes/{0}/spectra/x={1}/y={2}/'.format(self.mangaid, x, y)
            results = Interaction(route, request_type='get')
            return results.getData(astype=np.array)

    def _openFile(self):
        try:
            self.hdu = fits.open(self.filename)
        except FileNotFoundError as e:
            raise Exception('{0} does not exist. Please provide full file path.'.format(self.filename)) from e

    @property
    def flux(self):
        if config.mode == 'file':
            return self.hdu['FLUX'].data
        if config.mode == 'db':
            return None

    # Switch to _getCubeFromPlateIFU
    def _getCubeFromMangaID(self):
        ''' server-side code '''
        from ..db.database import db
        import sdss.internal.database.utah.mangadb.DataModelClasses as datadb
        session = db.Session()
        self._cube = session.query(datadb.Cube).join(datadb.PipelineInfo, datadb.PipelineVersion).filter(datadb.PipelineVersion.version == 'trunk', datadb.Cube.mangaid == self.mangaid).first()
        print('cube', self._cube, len(self._cube.spaxels))
        self.ifu = self._cube.ifu.name
        self.ra = self._cube.ra
        self.dec = self._cube.dec
        self.plate = self._cube.plate
