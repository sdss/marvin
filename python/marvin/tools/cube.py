from __future__ import print_function
import numpy as np
import os
from astropy.io import fits
from astropy import wcs
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

        self._useDB = None

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
                    # self._getCubeFromMangaID()
                    raise RuntimeError('Failed to connect to database.')
                except RuntimeError:
                    # convert from plateifu to filename (sdss_access)
                    # self.filename = plateifu_to_filename(self.plateifu)
                    plateifu_to_filename = {'8485-1901': 'test.fits'}
                    self.filename = plateifu_to_filename[self.plateifu]
                    try:
                        self._openFile()
                    except FileNotFoundError:
                        if config.download:
                            # download file via sdsssync
                            # self._openFile()
                            pass
                        else:
                            fnferr_msg = ('Failed to find file locally. Try downloading file.')
                            raise FileNotFoundError(fnferr_msg)
        elif config.mode == 'remote':
            if self.filename:
                raise FileNotFoundError('Cannot open {} remotely.'.format(self.filename))
            else:
                # use API
                pass

    def getSpectrum(self, x=None, y=None, ra=None, dec=None, ext='flux'):
        """Returns the appropriate spectrum for a certain spaxel in the cube.

        The type of the spectrum returned depends on the `ext` keyword, and
        may be either `'flux'`, `'ivar'`, or `'mask'`. The coordinates of the
        spectrum to return can be input as `x, y` pixels relative the centre
        of the cube (bottom-left origin is assumed), or as `ra, dec` celestial
        coordinates.

        Parameters
        ----------
        x, y : int
            The spaxel coordinates relative to the centre of the cube.

        ra, dec : float
            The coordinates of the spaxel to return. The closest spaxel to
            those coordinates will be returned.

        ext : str
            The extension of the cube to use, either `'flux'`, `'ivar'`, or
            `'mask'`.

        Returns
        -------
        result : np.array
            A Numpy array with the spectrum for the input coordinates.

        """

        # Checks that we have the correct set of inputs.
        if x or y:
            assert not ra and not dec, 'Either use (x, y) or (ra, dec)'
            assert x and y, 'Specify both x and y'
            inputMode = 'pix'
        elif ra or dec:
            assert not x and not y, 'Either use (x, y) or (ra, dec)'
            assert ra and dec, 'Specify both ra and dec'
            inputMode = 'sky'
        else:
            raise ValueError('You need to specify either (x, y) or (ra, dec)')

        assert isinstance(ext, basestring)
        ext = ext.lower()
        assert ext in ['flux', 'ivar', 'mask'], 'ext needs to be either \'flux\', \'ivar\', or \'mask\''

        if config.mode == 'local':

            if not self._useDB:

                cubeExt = self._hdu[ext.upper()]
                cubeShape = cubeExt.data.shape

                if inputMode == 'sky':
                    cubeWCS = wcs.WCS(cubeExt.header)
                    xCube, yCube, __ = cubeWCS.wcs_world2pix([[ra, dec, 1.]], 1)[0]
                else:
                    yMid, xMid = np.array(cubeShape[1:]) / 2.
                    xCube = int(xMid + x)
                    yCube = int(yMid - y)

                assert xCube > 0 and yCube > 0, 'pixel coordinates outside cube'
                assert (xCube < cubeShape[2] - 1 and yCube < cubeShape[1] - 1), 'pixel coordinates outside cube'

                return cubeExt.data[:, np.round(yCube), np.round(xCube)]

            else:

                raise NotImplementedError('getSpectrum from DB not yet implemented')

        else:

            raise NotImplementedError('getSpectrum over API not yet implemented')

    def _openFile(self):

        if not os.path.exists(self.filename):
            raise ValueError('filename {0} cannot be found'.format(self.filename))

        self._useDB = False

    @property
    def flux(self):
        if not self._useDB:
            return self._hdu['FLUX'].data
        else:
            return None

    def _getCubeFromDB(self):
        ''' server-side code '''

        # look for drpver
        if not config.drpver:
            raise RuntimeError('drpver not set in config!')

        # parse the plate-ifu
        if self.plateifu:
            plate, ifu = self.plateifu.split('-')

        if not config.db:
            raise RuntimeError('No db connected')
        else:
            from marvin.utils.db.dbutils import testDbConnection
            import sqlalchemy
            if testDbConnection(config.session):
                datadb = config.datadb
                session = config.session
                self._cube = None
                try:
                    self._cube = session.query(datadb.Cube).join(datadb.PipelineInfo, datadb.PipelineVersion, datadb.IFUDesign).filter(datadb.PipelineVersion.version == config.drpver, datadb.Cube.plate == plate, datadb.Cube.ifu.name == ifu).one()
                except sqlalchemy.orm.exc.MultipleResultsFound as e:
                    raise RuntimeError('Could not retrieve cube for plate-ifu {0}: Multiple Results Found: {1}'.format(self.plateifu, e))
                except sqlalchemy.orm.exc.NoResultFound as e:
                    raise RuntimeError('Could not retrieve cube for plate-ifu {0}: No Results Found: {1}'.format(self.plateifu, e))
                except Exception as e:
                    raise RuntimeError('Could not retrieve cube for plate-ifu {0}: Unknown exception: {1}'.format(self.plateifu, e))

                if self._cube:
                    self.ifu = self._cube.ifu.name
                    self.ra = self._cube.ra
                    self.dec = self._cube.dec
                    self.plate = self._cube.plate
