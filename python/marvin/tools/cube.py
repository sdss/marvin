#!/usr/bin/env python2
# encoding: utf-8
#
# cube.py


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import warnings

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

import marvin
import marvin.core.exceptions
import marvin.tools.spaxel
import marvin.tools.maps
import marvin.utils.general.general

from marvin.core.core import MarvinToolsClass
from marvin.core.exceptions import MarvinError, MarvinUserWarning

try:
    import photutils.aperture_funcs
except ImportError:
    photutils = False


class Cube(MarvinToolsClass):
    """A class to interface with MaNGA DRP data cubes.

    This class represents a fully reduced DRP data cube, initialised either
    from a file, a database, or remotely via the Marvin API.

    Parameters:
        data (``HDUList``, SQLAlchemy object, or None):
            An astropy ``HDUList`` or a SQLAlchemy object of a cube, to
            be used for initialisation. If ``None``, the normal mode will
            be used (see :ref:`mode-decision-tree`).
        filename (str):
            The path of the file containing the data cube to load.
        mangaid (str):
            The mangaid of the data cube to load.
        plateifu (str):
            The plate-ifu of the data cube to load (either ``mangaid`` or
            ``plateifu`` can be used, but not both).
        nsa_source ({'auto', 'drpall', 'nsa'}):
            Defines how the NSA data for this object should loaded when
            ``Cube.nsa`` is first called. If ``drpall``, the drpall file will
            be used (note that this will only contain a subset of all the NSA
            information); if ``nsa``, the full set of data from the DB will be
            retrieved. If the drpall file or a database are not available, a
            remote API call will be attempted. If ``nsa_source='auto'``, the
            source will depend on how the ``Cube`` object has been
            instantiated. If the cube has ``Cube.data_origin='file'``,
            the drpall file will be used (as it is more likely that the user
            has that file in their system). Otherwise, ``nsa_source='nsa'``
            will be assumed. This behaviour can be modified during runtime by
            modifying the ``Cube.nsa_mode`` with one of the valid values.
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See :ref:`mode-decision-tree`.
        release (str):
            The MPL/DR version of the data to use.

    Return:
        cube:
            An object representing the data cube.

    """

    def __init__(self, *args, **kwargs):

        valid_kwargs = [
            'data', 'filename', 'mangaid', 'plateifu', 'mode', 'release',
            'drpall', 'nsa_source']

        assert len(args) == 0, 'Cube does not accept arguments, only keywords.'
        for kw in kwargs:
            assert kw in valid_kwargs, 'keyword {0} is not valid'.format(kw)

        self.shape = None
        self.wcs = None
        self.wavelength = None
        self._drpall_data = None

        super(Cube, self).__init__(*args, **kwargs)

        if self.data_origin == 'file':
            self._load_cube_from_file(data=self.data)
        elif self.data_origin == 'db':
            self._load_cube_from_db(data=self.data)
        elif self.data_origin == 'api':
            self._load_cube_from_api()

        self._init_attributes()

        # Checks that the drpver set in MarvinToolsClass matches the header
        header_drpver = self.header['VERSDRP3'].strip()
        header_drpver = 'v1_5_1' if header_drpver == 'v1_5_0' else header_drpver
        assert header_drpver == self._drpver, ('mismatch between cube._drpver={0} '
                                               'and header drpver={1}'.format(self._drpver,
                                                                              header_drpver))

    def _getFullPath(self):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return super(Cube, self)._getFullPath('mangacube', ifu=ifu,
                                              drpver=self._drpver, plate=plate)

    def download(self):
        """Downloads the cube using sdss_access - Rsync,"""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return super(Cube, self).download('mangacube', ifu=ifu,
                                          drpver=self._drpver, plate=plate)

    def __repr__(self):
        """Representation for Cube."""

        return ('<Marvin Cube (plateifu={0}, mode={1}, data_origin={2})>'
                .format(repr(self.plateifu), repr(self.mode),
                        repr(self.data_origin)))

    def __getitem__(self, xy):
        """Returns the spaxel for ``(x, y)``"""

        return self.getSpaxel(x=xy[0], y=xy[1], xyorig='lower')

    def _init_attributes(self):
        """Initialises several attributes."""

        self.ra = float(self.header['OBJRA'])
        self.dec = float(self.header['OBJDEC'])

        self.plate = int(self.header['PLATEID'])
        self.ifu = int(self.header['IFUDSGN'])
        self.mangaid = self.header['MANGAID']

        self._isbright = 'APOGEE' in self.header['SRVYMODE']

        self.dir3d = 'mastar' if self._isbright else 'stack'

    def _load_cube_from_file(self, data=None):
        """Initialises a cube from a file."""

        if data is not None:
            assert isinstance(data, fits.HDUList), 'data is not an HDUList object'
        else:
            try:
                self.data = fits.open(self.filename)
            except IOError as err:
                raise IOError('filename {0} cannot be found: {1}'.format(self.filename, err))

        self.header = self.data[1].header
        self.shape = self.data['FLUX'].data.shape[1:]
        self.wcs = WCS(self.header)
        self.wavelength = self.data['WAVE'].data
        self.plateifu = self.header['PLATEIFU']

        # Checks and populates the release.
        file_drpver = self.header['VERSDRP3']
        file_drpver = 'v1_5_1' if file_drpver == 'v1_5_0' else file_drpver

        file_ver = marvin.config.lookUpRelease(file_drpver)
        assert file_ver is not None, 'cannot find file version.'

        if file_ver != self._release:
            warnings.warn('mismatch between file release={0} and object release={1}. '
                          'Setting object release to {0}'.format(file_ver, self._release),
                          MarvinUserWarning)
            self._release = file_ver

        self._drpver, self._dapver = marvin.config.lookUpVersions(release=self._release)

    def _load_cube_from_db(self, data=None):
        """Initialises a cube from the DB."""

        mdb = marvin.marvindb
        plate, ifu = self.plateifu.split('-')

        if not mdb.isdbconnected:
            raise MarvinError('No db connected')
        else:
            import sqlalchemy
            datadb = mdb.datadb

            if self.data:
                assert isinstance(data, datadb.Cube), 'data is not an instance of mangadb.Cube.'
                self.data = data
            else:
                try:
                    self.data = mdb.session.query(datadb.Cube).join(
                        datadb.PipelineInfo,
                        datadb.PipelineVersion,
                        datadb.IFUDesign).filter(
                            mdb.datadb.PipelineVersion.version == self._drpver,
                            datadb.Cube.plate == int(plate), datadb.IFUDesign.name == ifu).one()
                except sqlalchemy.orm.exc.MultipleResultsFound as ee:
                    raise MarvinError('Could not retrieve cube for plate-ifu {0}: '
                                      'Multiple Results Found: {1}'.format(self.plateifu, ee))
                except sqlalchemy.orm.exc.NoResultFound as ee:
                    raise MarvinError('Could not retrieve cube for plate-ifu {0}: '
                                      'No Results Found: {1}'.format(self.plateifu, ee))
                except Exception as ee:
                    raise MarvinError('Could not retrieve cube for plate-ifu {0}: '
                                      'Unknown exception: {1}'.format(self.plateifu, ee))

            self.header = self.data.header
            self.wcs = WCS(self.data.wcs.makeHeader())
            self.data = self.data
            self.shape = self.data.shape.shape
            self.wavelength = self.data.wavelength.wavelength

    def _load_cube_from_api(self):
        """Calls the API and retrieves the necessary information to instantiate the cube."""

        url = marvin.config.urlmap['api']['getCube']['url']

        try:
            response = self._toolInteraction(url.format(name=self.plateifu))
        except Exception as ee:
            raise MarvinError('found a problem when checking if remote cube '
                              'exists: {0}'.format(str(ee)))

        data = response.getData()

        self.header = fits.Header.fromstring(data['header'])
        self.shape = data['shape']
        self.wavelength = np.array(data['wavelength'])
        self.wcs = WCS(fits.Header.fromstring(data['wcs_header']))

        if self.plateifu != data['plateifu']:
            raise MarvinError('remote cube has a different plateifu!')

        return

    def _getExtensionData(self, extName):
        """Returns the data from an extension."""

        if self.data_origin == 'file':
            return self.data[extName.upper()].data
        elif self.data_origin == 'db':
            return self.data.get3DCube(extName.lower())
        elif self.data_origin == 'api':
            raise MarvinError('this feature does not work in remote mode. Use getSpaxel()')

    flux = property(lambda self: self._getExtensionData('FLUX'),
                    doc='Gets the `FLUX` data extension.')
    ivar = property(lambda self: self._getExtensionData('IVAR'),
                    doc='Gets the `IVAR` data extension.')
    mask = property(lambda self: self._getExtensionData('MASK'),
                    doc='Gets the `MASK` data extension.')

    @property
    def qualitybit(self):
        """The Cube DRP3QUAL bits."""

        # Python 2-3 compatibility
        try:
            bit = long(self.header['DRP3QUAL'])
        except NameError:
            bit = int(self.header['DRP3QUAL'])

        labels = None

        # get labels
        if self.data_origin == 'db':
            labels = self.data.getQualFlags()
        elif self.data_origin == 'file':
            pass
        elif self.data_origin == 'api':
            pass

        return 'DRP3QUAL', bit, labels

    @property
    def targetbit(self):
        """The Cube MNGTRG bits."""

        try:
            names = ['MNGTARG1', 'MNGTARG2', 'MNGTARG3']
            targs = [int(self.header[names[0]]), int(self.header[names[1]]),
                     int(self.header[names[2]])]
        except KeyError:
            names = ['MNGTRG1', 'MNGTRG2', 'MNGTRG3']
            targs = [int(self.header[names[0]]), int(self.header[names[1]]),
                     int(self.header[names[2]])]

        ind = np.nonzero(targs)[0]

        finaltargs = {}
        finaltargs['names'] = [names[i] for i in ind]
        finaltargs['bits'] = [targs[i] for i in ind]

        # get labels
        if self.data_origin == 'db':
            finaltargs['labels'] = [self.data.getTargFlags(type=i + 1) for i in ind]
        elif self.data_origin == 'file':
            pass
        elif self.data_origin == 'api':
            pass

        return finaltargs

    def getSpaxel(self, x=None, y=None, ra=None, dec=None,
                  properties=True, modelcube=False, **kwargs):
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
            properties (bool):
                If ``True``, the spaxel will be initiated with the DAP
                properties from the default Maps matching this cube.
            modelcube (:class:`~marvin.tools.modelcube.ModelCube` or None or bool):
                A :class:`~marvin.tools.modelcube.ModelCube` object
                representing the DAP modelcube entity. If None, the |spaxel|
                will be returned without model information. Default is False.

        Returns:
            spaxels (list):
                The |spaxel| objects for this cube corresponding to the input
                coordinates. The length of the list is equal to the number
                of input coordinates.

        .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

        """

        # TODO: do we want to use x/y, ra/dec, or a single coords parameter (as
        # an array of coordinates) and a mode keyword.

        kwargs['cube'] = self
        kwargs['maps'] = properties
        kwargs['modelcube'] = modelcube

        return marvin.utils.general.general.getSpaxel(x=x, y=y, ra=ra, dec=dec, **kwargs)

    def getMaps(self, **kwargs):
        """Retrieves the DAP :class:`~marvin.tools.maps.Maps` for this cube.

        If called without additional ``kwargs``, :func:`getMaps` will initilise
        the :class:`~marvin.tools.maps.Maps` using the ``plateifu`` of this
        :class:`~marvin.tools.cube.Cube`. Otherwise, the ``kwargs`` will be
        passed when initialising the :class:`~marvin.tools.maps.Maps`.

        """

        if len(kwargs.keys()) == 0 or 'filename' not in kwargs:
            kwargs.update({'plateifu': self.plateifu, 'release': self._release})

        maps = marvin.tools.maps.Maps(**kwargs)
        maps._cube = self
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
                'elliptical apertures are not yet implemented')

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
