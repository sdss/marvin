from __future__ import print_function
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import marvin
from marvin.tools.core import MarvinToolsClass
from marvin.tools.core.exceptions import MarvinError
from marvin.utils.general import convertCoords
from marvin.tools.spaxel import Spaxel
from marvin.api.api import Interaction


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
        skip_check (bool):
            If True, and ``mode='remote'``, skips the API call to check that
            the cube exists.
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
                                              drpver=self._drpver,
                                              plate=plate)

    def __init__(self, *args, **kwargs):

        self.filename = None
        self._hdu = None
        self._cube = None

        skip_check = kwargs.get('skip_check', False)

        super(Cube, self).__init__(*args, **kwargs)

        if self.data_origin == 'file':
            try:
                self._openFile()
            except IOError as e:
                raise MarvinError('Could not initialize via filename: {0}'.format(e))
        elif self.data_origin == 'db':
            try:
                self._getCubeFromDB()
            except RuntimeError as e:
                raise MarvinError('Could not initialize via db: {0}'.format(e))
        elif self.data_origin == 'api':
            if not skip_check:
                self._checkCubeRemote()

    def __repr__(self):
        """Representation for Cube."""

        return ('<Marvin Cube (plateifu={0}, mode={1}, data_origin={2})>'
                .format(repr(self.plateifu), repr(self.mode),
                        repr(self.data_origin)))

    def getSpaxel(self, x=None, y=None, ra=None, dec=None, xyorig='center'):
        """Returns the |spaxel| matching certain coordinates.

        The coordinates of the spaxel to return can be input as ``x, y`` pixels
        relative to``xyorig`` in the cube, or as ``ra, dec`` celestial
        coordinates.

        Parameters:
            x,y (int or array):
                The spaxel coordinates relative to ``xyorig``. If ``x`` is an
                array of coordinates, the size of ``x`` must much that of
                ``y``.
            ra,dec (float or array):
                The coordinates of the spaxel to return. The closest spaxel to
                those coordinates will be returned. If ``ra`` is an array of
                coordinates, the size of ``ra`` must much that of ``dec``.
            xyorig ({'center', 'lower'}):
                The reference point from which ``x`` and ``y`` are measured.
                Valid values are ``'center'`` (default), for the centre of the
                spatial dimensions of the cube, or ``'lower'`` for the
                lower-left corner. This keyword is ignored if ``ra`` and
                ``dec`` are defined.

        Returns:
            spaxels (list):
                The |spaxel| objects for this cube corresponding to the input
                coordinates. The length of the list is equal to the number
                of input coordinates.

        .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

        """

        # TBD: do we want to use x/y, ra/dec, or a single coords parameter (as
        # an array of coordinates) and a mode keyword.

        # Checks that we have the correct set of inputs.
        if x is not None or y is not None:
            assert ra is None and dec is None, 'Either use (x, y) or (ra, dec)'
            assert x is not None and y is not None, 'Specify both x and y'

            inputMode = 'pix'
            isScalar = np.isscalar(x)
            x = np.atleast_1d(x)
            y = np.atleast_1d(y)
            coords = np.array([x, y], np.float).T

        elif ra is not None or dec is not None:
            assert x is None and y is None, 'Either use (x, y) or (ra, dec)'
            assert ra is not None and dec is not None, 'Specify both ra and dec'

            inputMode = 'sky'
            isScalar = np.isscalar(ra)
            ra = np.atleast_1d(ra)
            dec = np.atleast_1d(dec)
            coords = np.array([ra, dec], np.float).T

        else:
            raise ValueError('You need to specify either (x, y) or (ra, dec)')

        if not xyorig:
            xyorig = 'center'

        if self.data_origin == 'file':

            # Uses the flux extension to get the WCS
            cubeExt = self._hdu['FLUX']
            cubeShape = cubeExt.data.shape[1:]

            ww = WCS(cubeExt.header) if inputMode == 'sky' else None

            iCube, jCube = zip(convertCoords(coords, wcs=ww, shape=cubeShape, mode=inputMode,
                                             xyorig=xyorig).T)

            _spaxels = []
            for ii in range(len(iCube[0])):
                _spaxels.append(
                    Spaxel._initFromData(jCube[0][ii], iCube[0][ii], self._hdu))

        elif self.data_origin == 'db':

            size = int(np.sqrt(len(self._cube.spaxels)))
            cubeShape = (size, size)

            if inputMode == 'sky':
                cubehdr = self._cube.wcs.makeHeader()
                ww = WCS(cubehdr)
            else:
                ww = None

            iCube, jCube = zip(convertCoords(coords, wcs=ww, shape=cubeShape, mode=inputMode,
                                             xyorig=xyorig).T)

            _spaxels = []
            for ii in range(len(iCube[0])):
                _spaxels.append(Spaxel(jCube[0][ii], iCube[0][ii], plateifu=self.plateifu))

        elif self.data_origin == 'api':

            path = '{0}={1}/{2}={3}/xyorig={4}'.format(
                'x' if inputMode == 'pix' else 'ra', coords[:, 0].tolist(),
                'y' if inputMode == 'pix' else 'dec', coords[:, 1].tolist(), xyorig)

            routeparams = {'name': self.plateifu, 'path': path}

            # Get the getSpaxel route
            url = marvin.config.urlmap['api']['getspaxels']['url'].format(**routeparams)

            response = Interaction(url)
            data = response.getData()

            xx = data['x']
            yy = data['y']

            _spaxels = []
            for ii in range(len(xx)):
                _spaxels.append(Spaxel(xx[ii], yy[ii], plateifu=self.plateifu, mode='remote'))

        if len(_spaxels) == 1 and isScalar:
            return _spaxels[0]
        else:
            return _spaxels

    def _openFile(self):
        """Initialises a cube from a file."""

        self._useDB = False
        try:
            self._hdu = fits.open(self.filename)
        except IOError as err:
            raise IOError('IOError: Filename {0} cannot be found: {1}'.format(self.filename, err))

        self.mangaid = self._hdu[0].header['MANGAID'].strip()
        self.plateifu = self._hdu[0].header['PLATEIFU'].strip()

    def _checkCubeRemote(self):
        """Calls the API to check that the cube exists."""

        url = marvin.config.urlmap['api']['getCube']['url']

        try:
            response = Interaction(url.format(name=self.plateifu))
        except Exception as ee:
            raise MarvinError('found a problem when checking if remote cube '
                              'exists: {0}'.format(str(ee)))

        data = response.getData()

        if self.plateifu not in data:
            raise MarvinError('remote cube has a different plateifu!')

        return

    def __getitem__(self, xy):
        """Returns the spaxel for ``(x, y)``"""
        x, y = xy
        return self.getSpaxel(x=x, y=y, xyorig='lower')

    def _getExtensionData(self, extName):
        """Returns the data from an extension."""

        if self.data_origin == 'file':
            return self._hdu[extName.upper()].data
        elif self.data_origin == 'db':
            return self._cube.get3dCube(extName.lower())
        elif self.data_origin == 'api':
            raise MarvinError('this feature does not work in remote mode. Use getSpaxel()')

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
