from __future__ import print_function
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import marvin
from marvin.tools.core import MarvinToolsClass, MarvinError
from marvin.utils.general import convertCoords, getSpaxelXY, getSpaxelAPI


class Cube(MarvinToolsClass):
    """A class to interface with MaNGA DRP data cubes.

    This class represents a fully reduced DRP data cube, initialised either
    from a file, a database, or remotely via the Marvin API.

    Parameters:
        filename (str):
            The path of the file containing the data cube to load.
        mangaid (str):
            The mangaid of the data cube to load.
        plateifu (str):
            The plate-ifu of the data cube to load (either ``mangaid`` or
            ``plateifu`` can be used, but not both).
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See
            :doc:`Mode secision tree</mode_decision>`.
        drpall (str):
            The path to the drpall file to use. Defaults to
            ``marvin.config.drpall``.
        drpver (str):
            The DRP version to use. Defaults to ``marvin.config.drpver``.

    Return:
        cube:
            An object representing the data cube.

    """

    def _getFullPath(self, **kwargs):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return super(Cube, self)._getFullPath('mangacube', ifu=ifu,
                                              drpver=marvin.config.drpver,
                                              plate=plate)

    def __init__(self, *args, **kwargs):

        super(Cube, self).__init__(*args, **kwargs)

        if self.mode == 'local':
            if self.filename:
                try:
                    self._openFile()
                except IOError as e:
                    raise MarvinError('Could not initialize via filename: {0}'.format(e))
            else:
                try:
                    self._getCubeFromDB()
                except RuntimeError as e:
                    raise MarvinError('Could not initialize via db: {0}'.format(e))
        else:
            # Placeholder for potentially request something from the DB to
            # initialise the cube.
            # raise MarvinError('Should remotely grab the cube to initialize, '
            #                   'but I won't')

            # TBD: For now we just pass. In the future, it would be good to
            # run a check with the API that the input parameters correspond to
            # a cube.
            pass

    def getSpectrum(self, x=None, y=None, ra=None, dec=None, ext=None,
                    xyorig='center'):
        """Returns the appropriate spectrum for a certain spaxel in the cube.

        The type of the spectrum returned depends on the `ext` keyword, and
        may be either ``'flux'``, `'ivar'`, or ``'mask'``. The coordinates of
        the spectrum to return can be input as ``x, y`` pixels relative to
        ``xyorig`` in the cube, or as ``ra, dec`` celestial coordinates.

        Parameters:
            x,y (int or array):
                The spaxel coordinates relative to ``xyorig``. If ``x`` is an
                array of coordinates, the size of ``x`` must much that of
                ``y``.

            ra,dec (float or array):
                The coordinates of the spaxel to return. The closest spaxel to
                those coordinates will be returned. If ``ra`` is an array of
                coordinates, the size of ``ra`` must much that of ``dec``.

            ext ({'flux', 'ivar', 'flux'}):
                The extension of the cube to use. Defaults to ``'flux'``.

            xyorig ({'center', 'lower'}):
                The reference point from which ``x`` and ``y`` are measured.
                Valid values are ``'center'`` (default), for the centre of the
                spatial dimensions of the cube, or ``'lower'`` for the
                lower-left corner. This keyword is ignored if ``ra`` and
                ``dec`` are defined.

        Returns:
            spectra (Numpy array):
                A Numpy array with the spectrum for the input coordinates.
                If the input coordinates are an array of N positions, the
                returned array will have shape (N, M) where M is the number of
                spectral elements.

        """

        # TBD: do we want to use x/y, ra/dec, or a single coords parameter (as
        # an array of coordinates) and a mode keyword.

        # Checks that we have the correct set of inputs.
        if x is not None or y is not None:
            assert ra is None and dec is None, 'Either use (x, y) or (ra, dec)'
            assert x is not None and y is not None, 'Specify both x and y'

            inputMode = 'pix'
            x = np.atleast_1d(x)
            y = np.atleast_1d(y)
            coords = np.array([x, y], np.float).T

        elif ra is not None or dec is not None:
            assert x is None and y is None, 'Either use (x, y) or (ra, dec)'
            assert ra is not None and dec is not None, 'Specify both ra and dec'

            inputMode = 'sky'
            ra = np.atleast_1d(ra)
            dec = np.atleast_1d(dec)
            coords = np.array([ra, dec], np.float).T

        else:
            raise ValueError('You need to specify either (x, y) or (ra, dec)')

        if not ext:
            ext = 'flux'

        if not xyorig:
            xyorig = 'center'

        try:
            isExtString = isinstance(ext, basestring)
        except NameError:
            isExtString = isinstance(ext, str)

        assert isExtString

        ext = ext.lower()
        assert ext in ['flux', 'ivar', 'mask'], 'ext needs to be either \'flux\', \'ivar\', or \'mask\''

        if self.mode == 'local':

            # Local mode

            if not self._useDB:

                # File mode

                cubeExt = self._hdu[ext.upper()]
                cubeShape = cubeExt.data.shape[1:]

                ww = WCS(cubeExt.header) if inputMode == 'sky' else None

                iCube, jCube = zip(convertCoords(coords, wcs=ww, shape=cubeShape, mode=inputMode, xyorig=xyorig).T)

                data = cubeExt.data[:, iCube[0], jCube[0]].T

            else:

                # DB mode

                size = int(np.sqrt(len(self._cube.spaxels)))
                cubeShape = (size, size)

                if inputMode == 'sky':
                    cubehdr = self._cube.wcs.makeHeader()
                    ww = WCS(cubehdr)
                else:
                    ww = None

                iCube, jCube = zip(convertCoords(coords, wcs=ww, shape=cubeShape, mode=inputMode, xyorig=xyorig).T)

                data = []
                for ii in range(len(iCube[0])):
                    spaxel = getSpaxelXY(self._cube, self.plateifu, x=jCube[0][ii], y=iCube[0][ii])
                    data.append(spaxel.__getattribute__(ext))

                data = np.array(data)

        else:

            # API mode

            # Fail if no route map initialized
            if not marvin.config.urlmap:
                raise MarvinError('No URL Map found. Cannot make remote call')

            data = []
            for ii in range(coords.shape[0]):
                spaxel = getSpaxelAPI(coords[ii][0], coords[ii][1], self.mangaid, mode=inputMode, ext=ext, xyorig=xyorig)
                data.append(spaxel)

            data = np.array(data)

        if data.shape[0] == 1:
            return data[0]
        else:
            return data

    def _openFile(self):

        self._useDB = False
        try:
            self._hdu = fits.open(self.filename)
        except IOError as err:
            raise IOError('IOError: Filename {0} cannot be found: {1}'.format(self.filename, err))

    def _getExtensionData(self, extName):
        """Returns the data from an extension."""

        if not self._useDB:
            return self._hdu[extName.upper()].data
        else:
            return None

    flux = property(lambda self: self._getExtensionData('FLUX'), doc='Gets the `FLUX` data extension.')
    ivar = property(lambda self: self._getExtensionData('IVAR'), doc='Gets the `IVAR` data extension.')
    mask = property(lambda self: self._getExtensionData('MASK'), doc='Gets the `MASK` data extension.')

    def getWavelength(self):
        ''' Retrieve the wavelength array for the Cube '''
        if self._useDB:
            if self._cube:
                wavelength = self._cube.wavelength.wavelength
            else:
                raise MarvinError('Cannot retrieve wavelength.  No DB cube entry found!')
        else:
            if self._hdu:
                wavelength = self._hdu['WAVE'].data
            else:
                raise MarvinError('Cannot retrieve wavelength.  No HDU found!')
        return wavelength

    def _getCubeFromDB(self):
        ''' server-side code '''

        mdb = marvin.marvindb

        # look for drpver
        if not marvin.config.drpver:
            raise RuntimeError('drpver not set in config!')

        # parse the plate-ifu
        if self.plateifu:
            plate, ifu = self.plateifu.split('-')

        if not mdb.isdbconnected:
            raise RuntimeError('No db connected')
        else:
            import sqlalchemy
            self._cube = None
            try:
                self._cube = mdb.session.query(mdb.datadb.Cube).join(mdb.datadb.PipelineInfo, mdb.datadb.PipelineVersion, mdb.datadb.IFUDesign).\
                    filter(mdb.datadb.PipelineVersion.version == marvin.config.drpver, mdb.datadb.Cube.plate == plate,
                           mdb.datadb.IFUDesign.name == ifu).one()
            except sqlalchemy.orm.exc.MultipleResultsFound as e:
                raise RuntimeError('Could not retrieve cube for plate-ifu {0}: Multiple Results Found: {1}'.format(self.plateifu, e))
            except sqlalchemy.orm.exc.NoResultFound as e:
                raise RuntimeError('Could not retrieve cube for plate-ifu {0}: No Results Found: {1}'.format(self.plateifu, e))
            except Exception as e:
                raise RuntimeError('Could not retrieve cube for plate-ifu {0}: Unknown exception: {1}'.format(self.plateifu, e))

            if self._cube:
                self._useDB = True
                self.hdr = self._cube.header
                self.wcs = self._cube.wcs.makeHeader()
                self.ifu = self._cube.ifu.name
                self.ra = self._cube.ra
                self.dec = self._cube.dec
                self.plate = self._cube.plate
                self.mangaid = self._cube.mangaid
                self.redshift = self._cube.sample[0].nsa_redshift  # TODO change this for the future to sampledb
