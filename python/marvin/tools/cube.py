#!/usr/bin/env python3
# encoding: utf-8
#
# cube.py


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

import marvin
import marvin.core.exceptions
import marvin.tools.spaxel
import marvin.tools.maps
import marvin.utils.general.general

from marvin.api.api import Interaction
from marvin.core import MarvinToolsClass
from marvin.core.exceptions import MarvinError

try:
    import photutils.aperture_funcs
except:
    photutils = False


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
            The load mode to use. See :doc:`mode-decision-tree>`.
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

    def download(self, **kwargs):
        ''' Downloads the cube using sdss_access - Rsync '''
        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return super(Cube, self).download('mangacube', ifu=ifu,
                                          drpver=self._drpver,
                                          plate=plate)

    def __init__(self, *args, **kwargs):

        # TODO: consolidate _hdu/_cube in data. This class needs a clean up.
        # Can use Maps or Spaxel as an example. For now I'm adding more
        # clutter to avoid breaking things (JSG).

        self._hdu = None
        self._cube = None
        self._shape = None

        self.filename = None
        self.wcs = None
        self.data = None
        self.wavelength = None

        skip_check = kwargs.get('skip_check', False)

        super(Cube, self).__init__(*args, **kwargs)

        if self.data_origin == 'file':
            try:
                self._openFile()
            except IOError as e:
                raise MarvinError('Could not initialize via filename: {0}'.format(e))
            self.plateifu = self.hdr['PLATEIFU'].strip()
            self.redshift = None

        elif self.data_origin == 'db':
            try:
                self._getCubeFromDB()
            except RuntimeError as e:
                raise MarvinError('Could not initialize via db: {0}'.format(e))
            nsaobjs = self._cube.target.NSA_objects if self._cube.target else None
            if nsaobjs:
                self.redshift = None if len(nsaobjs) > 1 else nsaobjs[0].z
            else:
                self.redshift = None

        elif self.data_origin == 'api':
            if not skip_check:
                self._openCubeRemote()

        self.ifu = int(self.hdr['IFUDSGN'])
        self.ra = float(self.hdr['OBJRA'])
        self.dec = float(self.hdr['OBJDEC'])
        self.plate = int(self.hdr['PLATEID'])
        self.mangaid = self.hdr['MANGAID']
        self._isbright = 'APOGEE' in self.hdr['SRVYMODE']
        self.dir3d = 'mastar' if self._isbright else 'stack'

    def __repr__(self):
        """Representation for Cube."""

        return ('<Marvin Cube (plateifu={0}, mode={1}, data_origin={2})>'
                .format(repr(self.plateifu), repr(self.mode),
                        repr(self.data_origin)))

    def getSpaxel(self, x=None, y=None, ra=None, dec=None, xyorig=None):
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
                Valid values are ``'center'``, for the centre of the
                spatial dimensions of the cube, or ``'lower'`` for the
                lower-left corner. This keyword is ignored if ``ra`` and
                ``dec`` are defined. ``xyorig`` defaults to
                ``marvin.config.xyorig.``

        Returns:
            spaxels (list):
                The |spaxel| objects for this cube corresponding to the input
                coordinates. The length of the list is equal to the number
                of input coordinates.

        .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

        """

        # TODO: do we want to use x/y, ra/dec, or a single coords parameter (as
        # an array of coordinates) and a mode keyword.

        # TODO: adapt to use marvin.general.general.getSpaxel.

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
            xyorig = marvin.config.xyorig

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
                    marvin.tools.spaxel.Spaxel._initFromData(
                        self.plateifu, jCube[0][ii], iCube[0][ii], cube=self))

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
                _spaxels.append(
                    marvin.tools.spaxel.Spaxel(jCube[0][ii], iCube[0][ii],
                                               plateifu=self.plateifu,
                                               drpver=self._drpver))

        elif self.data_origin == 'api':

            # TODO: we are doing two interactions for what probably can be
            # accomplished with one.

            path = '{0}={1}/{2}={3}/xyorig={4}'.format(
                'x' if inputMode == 'pix' else 'ra', coords[:, 0].tolist(),
                'y' if inputMode == 'pix' else 'dec', coords[:, 1].tolist(), xyorig)

            routeparams = {'name': self.plateifu, 'path': path}

            # Get the getSpaxel route
            url = marvin.config.urlmap['api']['getspaxels']['url'].format(**routeparams)

            response = Interaction(url, params={'drpver': self._drpver})
            data = response.getData()

            xx = data['x']
            yy = data['y']

            _spaxels = []
            for ii in range(len(xx)):
                _spaxels.append(
                    marvin.tools.spaxel.Spaxel(xx[ii], yy[ii],
                                               plateifu=self.plateifu,
                                               mode='remote',
                                               drpver=self._drpver))

        # Sets the shape of the cube on the spaxels
        for sp in _spaxels:
            sp._parent_shape = self.shape

        if len(_spaxels) == 1 and isScalar:
            return _spaxels[0]
        else:
            return _spaxels

    def _openFile(self):
        """Initialises a cube from a file."""

        self._useDB = False
        try:
            self._hdu = fits.open(self.filename)
            self.data = self._hdu
        except IOError as err:
            raise IOError('IOError: Filename {0} cannot be found: {1}'.format(self.filename, err))

        self.hdr = self._hdu[1].header
        self.wcs = WCS(self.hdr)
        self.wavelength = self._hdu['WAVE'].data

    def _openCubeRemote(self):
        """Calls the API to check that the cube exists and gets the header."""

        url = marvin.config.urlmap['api']['getCube']['url']

        try:
            response = Interaction(url.format(name=self.plateifu), params={'drpver': self._drpver})
        except Exception as ee:
            raise MarvinError('found a problem when checking if remote cube '
                              'exists: {0}'.format(str(ee)))

        data = response.getData()

        self.hdr = fits.Header.fromstring(data['header'])
        self.redshift = float(data['redshift'])
        self._shape = data['shape']
        self.wavelength = data['wavelength']
        self.wcs = WCS(fits.Header.fromstring(data['wcs_header']))

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
            return self._cube.get3DCube(extName.lower())
        elif self.data_origin == 'api':
            raise MarvinError('this feature does not work in remote mode. Use getSpaxel()')

    flux = property(lambda self: self._getExtensionData('FLUX'),
                    doc='Gets the `FLUX` data extension.')
    ivar = property(lambda self: self._getExtensionData('IVAR'),
                    doc='Gets the `IVAR` data extension.')
    mask = property(lambda self: self._getExtensionData('MASK'),
                    doc='Gets the `MASK` data extension.')

    @property
    def shape(self):
        """The shape of the cube."""

        if self._shape is None:
            if self.data_origin == 'file':
                self._shape = self._hdu['FLUX'].data.shape[1:]
            elif self.data_origin == 'db':
                self._shape = self._cube.shape.shape
            elif self.data_origin == 'api':
                # self._shape gets initialised in self._openCubeRemote
                pass

        return self._shape

    @property
    def qualitybit(self):
        ''' The Cube DRP3QUAL bits '''
        bit = long(self.hdr['DRP3QUAL'])
        labels = None
        # get labels
        if self.data_origin == 'db':
            labels = self._cube.getQualFlags()
        elif self.data_origin == 'file':
            pass
        elif self.data_origin == 'api':
            pass

        return 'DRP3QUAL', bit, labels

    @property
    def targetbit(self):
        ''' The Cube MNGTRG bits '''

        try:
            names = ['MNGTARG1', 'MNGTARG2', 'MNGTARG3']
            targs = [long(self.hdr[names[0]]), long(self.hdr[names[1]]), long(self.hdr[names[2]])]
        except KeyError as e:
            names = ['MNGTRG1', 'MNGTRG2', 'MNGTRG3']
            targs = [long(self.hdr[names[0]]), long(self.hdr[names[1]]), long(self.hdr[names[2]])]

        ind = np.nonzero(targs)[0]
        labels = None
        finaltargs = {}

        finaltargs['names'] = [names[i] for i in ind]
        finaltargs['bits'] = [targs[i] for i in ind]
        # get labels
        if self.data_origin == 'db':
            finaltargs['labels'] = [self._cube.getTargFlags(type=i+1) for i in ind]
        elif self.data_origin == 'file':
            pass
        elif self.data_origin == 'api':
            pass

        return finaltargs

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
                self._cube = mdb.session.query(mdb.datadb.Cube).join(mdb.datadb.PipelineInfo,
                                                                     mdb.datadb.PipelineVersion,
                                                                     mdb.datadb.IFUDesign).\
                    filter(mdb.datadb.PipelineVersion.version == self._drpver,
                           mdb.datadb.Cube.plate == plate,
                           mdb.datadb.IFUDesign.name == ifu).one()
            except sqlalchemy.orm.exc.MultipleResultsFound as e:
                raise RuntimeError('Could not retrieve cube for plate-ifu {0}: '
                                   'Multiple Results Found: {1}'.format(self.plateifu, e))
            except sqlalchemy.orm.exc.NoResultFound as e:
                raise RuntimeError('Could not retrieve cube for plate-ifu {0}: '
                                   'No Results Found: {1}'.format(self.plateifu, e))
            except Exception as e:
                raise RuntimeError('Could not retrieve cube for plate-ifu {0}: '
                                   'Unknown exception: {1}'.format(self.plateifu, e))

            if self._cube:
                self._useDB = True

                # TODO: this is ugly at so many levels ...
                self.hdr = fits.Header(eval(self._cube.hdr[0].header).items())

                self.wcs = WCS(self._cube.wcs.makeHeader())
                self.data = self._cube
                self.wavelength = self.data.wavelength.wavelength
            else:
                self._useDB = False

    def getMaps(self, **kwargs):
        """Retrieves the DAP :class:`~marvin.tools.maps.Maps` for this cube.

        If called without additional ``kwargs``, :func:`getMaps` will initilise
        the :class:`~marvin.tools.maps.Maps` using the ``plateifu`` of this
        :class:`~marvin.tools.cube.Cube`. Otherwise, the ``kwargs`` will be
        passed when initialising the :class:`~marvin.tools.maps.Maps`.

        """

        if len(kwargs.keys()) == 0 or 'filename' not in kwargs:
            kwargs.update({'plateifu': self.plateifu})

        maps = marvin.tools.maps.Maps(**kwargs)

        return maps

    def getAperture(self, coords, radius, mode='pix', weight=True,
                    return_type='mask'):
        """Returns the spaxel in a circular or elliptical aperture.

        Returns either a mask of the same shape as the cube with the spaxels
        within an aperture, or the integrated spaxel from combining the spectra
        for those spaxels.

        The centre of the aperture is defined by ``coords``, which must be a
        tuple of ``(x,y)`` (if ``mode='pix'``) or ``(ra,dec)`` coordinates
        (if ``mode='sky'``). ``radius`` defines the radius of the circular
        aperture, or the parameters of the aperture ellipse.

        If ``weight=True``, the returned mask indicated the fraction of the
        spaxel encompassed by the aperture, ranging from 0 for spaxels not
        included to 1 for pixels totally included in the aperture. This
        weighting is used to return the integrated spaxel.

        Parameters:
            coords (tuple):
                Either the ``(x,y)`` or ``(ra,dec)`` coordinates of the centre
                of the aperture.
            radius (float or tuple):
                If a float, the radius of the circular aperture. If
                ``mode='pix'`` it must be the radius in pixels; if
                ``mode='sky'``, ``radius`` is in arcsec. To define an
                elliptical aperture, ``radius`` must be a 3-element tuple with
                the first two elements defining the major and minor semi-axis
                of the ellipse, and the third one the position angle in degrees
                from North to East.
            mode ({'pix', 'sky'}):
                Defines whether the values in ``coords`` and ``radius`` refer
                to pixels in the cube or angles on the sky.
            weight (bool):
                If ``True``, the returned mask or integrated spaxel will be
                weighted by the fractional pixels in the aperture.
            return_type ({'mask', 'mean', 'median', 'sum', 'spaxels'}):
                The type of data to be returned.

        Returns:
            result:
                If ``return_type='mask'``, this methods returns a 2D mask with
                the shape of the cube indicating the spaxels included in the
                aperture and, if appliable, their fractional contribution to
                the aperture. If ``spaxels``, both the mask (flattened to a
                1D array) and the :class:`~marvin.tools.spaxel.Spaxel`
                included in the aperture are returned. ``mean``, ``median``,
                or ``sum`` will allow arithmetic operations with the spaxels
                in the aperture in the future.

        Example:
            To get the mask for a circular aperture centred in spaxel (5, 7)
            and with radius 5 spaxels
              >>> mask = cube.getAperture((5, 7), 5)
              >>> mask.shape
              (34, 34)
            If you want to get the spaxels associated with that mask
              >>> mask, spaxels = cube.getAperture((5, 7), 5, return_type='spaxels')
              >>> len(spaxels)
              15

        """

        assert return_type in ['mask', 'mean', 'median', 'sum', 'spaxels']

        if return_type not in ['mask', 'spaxels']:
            raise marvin.core.exceptions.MarvinNotImplemented(
                'return_type={0} is not yet implemented'.format(return_type))

        if not photutils:
            raise MarvinError('getAperture currently requires photutils.')

        if mode != 'pix':
            raise marvin.core.exceptions.MarvinNotImplemented(
                'mode={0} is not yet implemented'.format(mode))

        if not np.isscalar(radius):
            raise marvin.core.exceptions.MarvinNotImplemented(
                'elliptical apertures are not yet implemented'.format(mode))

        data_mask = np.zeros(self.shape)

        if weight:
            phot_mode = ''
        else:
            phot_mode = 'center'

        coords = np.atleast_2d(coords) + 0.5

        mask = photutils.aperture_funcs.get_circular_fractions(
            data_mask, coords, radius, phot_mode, 0)

        if return_type == 'mask':
            return mask

        if return_type == 'spaxels':
            mask_idx = np.where(mask)
            spaxels = self.getSpaxel(x=mask_idx[0], y=mask_idx[1],
                                     xyorig='lower')

            fractions = mask[mask_idx]

            return (fractions, spaxels)
