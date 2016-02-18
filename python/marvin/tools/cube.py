from __future__ import print_function
import numpy as np
from astropy.io import fits
import marvin
from marvin.api.api import Interaction
from marvin.tools.core import MarvinToolsClass, MarvinError
from marvin.utils.general import convertCoords


class Cube(MarvinToolsClass):

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
            pass

    def getSpectrum(self, x=None, y=None, ra=None, dec=None, ext=None):
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
            `'mask'`. Defaults to `'flux'`.

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

        if not ext:
            ext = 'flux'

        try:
            isExtString = isinstance(ext, basestring)
        except NameError:
            isExtString = isinstance(ext, str)

        assert isExtString

        ext = ext.lower()
        assert ext in ['flux', 'ivar', 'mask'], 'ext needs to be either \'flux\', \'ivar\', or \'mask\''

        if self.mode == 'local':

            if not self._useDB:

                cubeExt = self._hdu[ext.upper()]
                cubeShape = cubeExt.data.shape

                if inputMode == 'sky':
                    xCube, yCube = convertCoords(ra=ra, dec=dec, hdr=cubeExt.header, mode='sky')
                else:
                    xCube, yCube = convertCoords(x=x, y=y, shape=cubeShape[1:], mode='pix')

                assert xCube > 0 and yCube > 0, 'pixel coordinates outside cube'
                assert (xCube < cubeShape[2] - 1 and yCube < cubeShape[1] - 1), 'pixel coordinates outside cube'

                return cubeExt.data[:, np.round(yCube), np.round(xCube)]

            else:

                if inputMode == 'sky':
                    cubehdr = self._cube.wcs.makeHeader()
                    xCube, yCube = convertCoords(ra=ra, dec=dec, hdr=cubehdr, mode='sky')
                else:
                    size = int(np.sqrt(len(self._cube.spaxels)))
                    shape = (size, size)
                    xCube, yCube = convertCoords(x=x, y=y, shape=shape, mode='pix')

                assert xCube > 0 and yCube > 0, 'pixel coordinates outside cube'

                import sqlalchemy
                inputs = [ra, dec] if inputMode == 'sky' else [x, y]
                try:
                    spaxel = marvin.session.query(marvin.datadb.Spaxel).filter_by(cube=self._cube, x=np.round(xCube), y=np.round(yCube)).one()
                except sqlalchemy.orm.exc.NoResultFound as e:
                    raise MarvinError('Could not retrieve spaxel for plate-ifu {0} at position {1},{2}: No Results Found: {3}'.format(self.plateifu, inputs[0], inputs[1], e))
                except Exception as e:
                    raise MarvinError('Could not retrieve cube for plate-ifu {0} at position {1},{2}: Unknown exception: {3}'.format(self.plateifu, inputs[0], inputs[1], e))

                data = spaxel.__getattribute__(ext)
                return data

        else:
            # Fail if no route map initialized
            if not marvin.config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Parse the variables into right frame
            path = 'x={0}/y={1}/ext={2}'.format(x, y, ext) if (x or y) else 'ra={0}/dec={1}/ext={2}'.format(ra, dec, ext)
            routeparams = {'name': self.mangaid, 'path': path}

            # Get the getSpectrum Route
            url = marvin.config.urlmap['api']['getspectra']['url'].format(**routeparams)

            # Make the API call
            if x or y:
                response = Interaction(url)
            elif ra or dec:
                response = Interaction(url)

            if response.status_code == 200:
                if response.results['status'] == 1:
                    return response.getData()
                else:
                    raise MarvinError('Could not retrieve spaxels remotely: {0}'.format(response.results['error']))
            else:
                raise MarvinError('Error retrieving response: Http status code {0}: {1}'.format(response.status_code, response.results['message']))

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

    flux = property(lambda self: self._getExtensionData('FLUX'))
    ivar = property(lambda self: self._getExtensionData('IVAR'))
    mask = property(lambda self: self._getExtensionData('MASK'))

    def _getCubeFromDB(self):
        ''' server-side code '''

        # look for drpver
        if not marvin.config.drpver:
            raise RuntimeError('drpver not set in config!')

        # parse the plate-ifu
        if self.plateifu:
            plate, ifu = self.plateifu.split('-')

        if not marvin.config.db:
            raise RuntimeError('No db connected')
        else:
            import sqlalchemy
            self._cube = None
            try:
                self._cube = marvin.session.query(marvin.datadb.Cube).join(marvin.datadb.PipelineInfo, marvin.datadb.PipelineVersion, marvin.datadb.IFUDesign).\
                    filter(marvin.datadb.PipelineVersion.version == marvin.config.drpver, marvin.datadb.Cube.plate == plate,
                           marvin.datadb.IFUDesign.name == ifu).one()
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
