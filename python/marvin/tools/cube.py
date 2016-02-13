from __future__ import print_function
import numpy as np
from astropy.io import fits
from astropy import wcs
from marvin import config
from marvin.api.api import Interaction
from marvin.tools.core import MarvinToolsClass


class Cube(MarvinToolsClass):

    def _getFullPath(self, **kwargs):

        """
        customised stuff to get the full path for this type of data based
        on input plateifu (and maybe other data in kwargs).

        super(Cube, self)._getFullPath()
        we do stuff with self._Path
        return fullpath
        """

        super(Cube, self)._getFullPath()

        plateifu = kwargs['plateifu']
        plate, ifu = plateifu.split('-')

        return self._Path().full('mangacube', ifu=ifu,
                                 drpver=config.drpver, plate=plate)

    def __init__(self, *args, **kwargs):

        myKwargs = self._kwargs

        self.mode = myKwargs.get('mode', None)
        self.filename = myKwargs.get('filename', None)
        self.mangaid = myKwargs.get('mangaid', None)
        self.plateifu = myKwargs.get('plateifu', None)

        if self.mode == 'local':
            if self.filename:
                self._openFile()
            else:
                self._getCubeFromDB()
        else:
            self.testAPI = Interaction('cubes/', request_type='get').getData()

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

            response = Interaction('/cubes/{0}/spectra'.format(self.mangaid))

    def _openFile(self):

        try:
            self._hdu = fits.open(self.filename)
        except IOError as err:
            if not err.args:
                err.args = ('',)
            err.args = err.args + ('filename {0} cannot be found'
                                   .format(self.filename),)
            raise

        self._useDB = False

    def _getExtensionData(self, extName):
        """Returns the data from an extension."""

        if not self._useDB:
            return self._hdu[extName.upper()].data
        else:
            return None

    flux = property(lambda self: self._getExtensionData('FLUX'))
    ivar = property(lambda self: self._getExtensionData('IVAR'))
    mask = property(lambda self: self._getExtensionData('MASK'))

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
            if testDbConnection(config.session)['good']:
                datadb = config.datadb
                session = config.session
                self._cube = None
                try:
                    self._cube = session.query(datadb.Cube).join(datadb.PipelineInfo, datadb.PipelineVersion, datadb.IFUDesign).filter(datadb.PipelineVersion.version == config.drpver, datadb.Cube.plate == plate, datadb.IFUDesign.name == ifu).one()
                except sqlalchemy.orm.exc.MultipleResultsFound as e:
                    raise RuntimeError('Could not retrieve cube for plate-ifu {0}: Multiple Results Found: {1}'.format(self.plateifu, e))
                except sqlalchemy.orm.exc.NoResultFound as e:
                    raise RuntimeError('Could not retrieve cube for plate-ifu {0}: No Results Found: {1}'.format(self.plateifu, e))
                except Exception as e:
                    raise RuntimeError('Could not retrieve cube for plate-ifu {0}: Unknown exception: {1}'.format(self.plateifu, e))

                if self._cube:
                    self._useDB = True
                    self.ifu = self._cube.ifu.name
                    self.ra = self._cube.ra
                    self.dec = self._cube.dec
                    self.plate = self._cube.plate
