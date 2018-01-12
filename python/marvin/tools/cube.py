#!/usr/bin/env python
# encoding: utf-8
#
# @Author: Brian Cherinka, José Sánchez-Gallego, Brett Andrews
# @Date: Oct 25, 2017
# @Filename: base.py
# @License: BSD 3-Clause
# @Copyright: Brian Cherinka, José Sánchez-Gallego, Brett Andrews


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

from marvin.core.core import MarvinToolsClass, NSAMixIn
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.tools.quantities import DataCube, Spectrum
from marvin.utils.datamodel.drp import datamodel
from marvin.utils.general import get_nsa_data, FuzzyDict
from marvin.utils.general.maskbit import get_manga_target


class Cube(MarvinToolsClass, NSAMixIn):
    """A class to interface with MaNGA DRP data cubes.

    This class represents a fully reduced DRP data cube, initialised either
    from a file, a database, or remotely via the Marvin API.

    See `~.MarvinToolsClass` for a list of parameters. In addition to the
    attributes defined `there <~.MarvinToolsClass>`, the following ones are
    also defined

    Attributes:
        header (`astropy.io.fits.Header`):
            The header of the datacube.
        ifu (int):
            The id of the IFU.
        ra,dec (float):
            Coordinates of the target.
        wcs (`astropy.wcs.WCS`):
            The WCS solution for this plate

    """

    def __init__(self, input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None,
                 drpall=None, download=None, nsa_source='auto'):

        self.header = None
        self.wcs = None
        self._wavelength = None
        self._shape = None

        # Stores data from extensions that have already been accessed, so that they
        # don't need to be retrieved again.
        self._extension_data = {}

        # Datacubes and spectra
        self._flux = None
        self._spectral_resolution = None
        self._spectral_resolution_prepixel = None
        self._dispersion = None
        self._dispersion_prepixel = None
        self._bitmasks = None

        MarvinToolsClass.__init__(self, input=input, filename=filename, mangaid=mangaid,
                                  plateifu=plateifu, mode=mode, data=data, release=release,
                                  drpall=drpall, download=download)

        NSAMixIn.__init__(self, nsa_source=nsa_source)

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

    def _set_datamodel(self):
        """Sets the datamodel for DRP."""

        self.datamodel = datamodel[self.release.upper()]
        self._bitmasks = datamodel[self.release.upper()].bitmasks

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
                .format(repr(self.plateifu), repr(self.mode), repr(self.data_origin)))

    def __getitem__(self, xy):
        """Returns the spaxel for ``(x, y)``"""

        return self.getSpaxel(x=xy[1], y=xy[0], xyorig='lower')

    def _init_attributes(self):
        """Initialises several attributes."""

        self.ra = float(self.header['OBJRA'])
        self.dec = float(self.header['OBJDEC'])

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
            except (IOError, OSError) as err:
                raise OSError('filename {0} cannot be found: {1}'.format(self.filename, err))

        self.header = self.data[1].header
        self.wcs = WCS(self.header)

        self._wavelength = self.data['WAVE'].data
        self._shape = (self.data['FLUX'].header['NAXIS2'],
                       self.data['FLUX'].header['NAXIS1'])

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

            # Reload NSA data from file version of drpall file
            self._drpver, self._dapver = marvin.config.lookUpVersions(release=self._release)
            self._drpall = marvin.config._getDrpAllPath(file_drpver)
            self.mangaid = self.header['MANGAID']

            nsa_source = 'drpall' if self.nsa_source == 'auto' else self.nsa_source

            self._nsa = None
            self._nsa = get_nsa_data(self.mangaid, mode='auto', source=nsa_source,
                                     drpver=self._drpver, drpall=self._drpall)

        self._drpver, self._dapver = marvin.config.lookUpVersions(release=self._release)

    def _load_cube_from_db(self, data=None):
        """Initialises a cube from the DB."""

        mdb = marvin.marvindb
        plate, ifu = self.plateifu.split('-')

        if not mdb.isdbconnected:
            raise MarvinError('No DB connected')
        else:
            import sqlalchemy
            datadb = mdb.datadb

            if self.data:
                assert isinstance(data, datadb.Cube), 'data is not an instance of mangadb.Cube.'
                self.data = data
            else:
                try:
                    self.data = mdb.session.query(datadb.Cube).join(
                        datadb.PipelineInfo, datadb.PipelineVersion, datadb.IFUDesign).filter(
                            mdb.datadb.PipelineVersion.version == self._drpver,
                            datadb.Cube.plate == int(plate),
                            datadb.IFUDesign.name == ifu).one()
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

            self._wavelength = np.array(self.data.wavelength.wavelength)
            self._shape = self.data.shape.shape

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
        self.wcs = WCS(fits.Header.fromstring(data['wcs_header']))
        self._wavelength = data['wavelength']
        self._shape = data['shape']

        if self.plateifu != data['plateifu']:
            raise MarvinError('remote cube has a different plateifu!')

        return

    def _get_datacube(self, name):
        """Returns a `.DataCube`."""

        model = self.datamodel.datacubes[name]
        cube_data = self._get_extension_data(name)

        if cube_data is None:
            raise MarvinError('cannot find data for this extension. '
                              'Maybe it is not loaded into the DB.')

        datacube = DataCube(cube_data,
                            np.array(self._wavelength),
                            ivar=self._get_extension_data(name, 'ivar'),
                            mask=self._get_extension_data(name, 'mask'),
                            unit=model.unit)

        return datacube

    def _get_spectrum(self, name):
        """Returns an `.Spectrum`."""

        model = self.datamodel.spectra[name]
        spec_data = self._get_extension_data(name)

        if spec_data is None:
            raise MarvinError('cannot find data for this extension. '
                              'Maybe it is not loaded into the DB.')

        spectrum = Spectrum(spec_data,
                            wavelength=np.array(self._wavelength),
                            std=self._get_extension_data(name, 'std'),
                            unit=model.unit)

        return spectrum

    @property
    def flux(self):
        """Returns a `.DataCube` object with the flux."""

        assert 'flux' in self.datamodel.datacubes.list_names(), \
            'flux is not present in his MPL version.'

        assert hasattr(self, '_flux')

        if self._flux is None:
            self._flux = self._get_datacube('flux')

        return self._flux

    @property
    def dispersion(self):
        """Returns a `.DataCube` object with the dispersion."""

        assert 'dispersion' in self.datamodel.datacubes.list_names(), \
            'dispersion is not present in his MPL version.'

        assert hasattr(self, '_dispersion')

        if self._dispersion is None:
            self._dispersion = self._get_datacube('dispersion')

        return self._dispersion

    @property
    def dispersion_prepixel(self):
        """Returns a `.DataCube` object with the prepixel dispersion."""

        assert 'dispersion_prepixel' in self.datamodel.datacubes.list_names(), \
            'dispersion_prepixel is not present in his MPL version.'

        assert hasattr(self, '_dispersion_prepixel')

        if self._dispersion_prepixel is None:
            self._dispersion_prepixel = self._get_datacube('dispersion_prepixel')

        return self._dispersion_prepixel

    @property
    def spectral_resolution(self):
        """Returns a `.Spectrum` with the spectral dispersion."""

        assert 'spectral_resolution' in self.datamodel.spectra.list_names(), \
            'spectral_resolution is not present in his MPL version.'

        assert hasattr(self, '_spectral_resolution')

        if self._spectral_resolution is None:
            self._spectral_resolution = self._get_spectrum('spectral_resolution')

        return self._spectral_resolution

    @property
    def spectral_resolution_prepixel(self):
        """Returns a `.Spectrum` with the prepixel spectral dispersion."""

        assert 'spectral_resolution_prepixel' in self.datamodel.spectra.list_names(), \
            'spectral_resolution_prepixel is not present in his MPL version.'

        assert hasattr(self, '_spectral_resolution_prepixel')

        if self._spectral_resolution_prepixel is None:
            self._spectral_resolution_prepixel = self._get_spectrum('spectral_resolution_prepixel')

        return self._spectral_resolution_prepixel

    def _get_ext_name(self, model, ext):
        ''' Get the extension name if it exists '''

        hasext = 'has_{0}'.format(ext)
        if hasattr(model, hasext):
            hasmeth = model.__getattribute__(hasext)()
            if not hasmeth:
                return None

        return model.fits_extension(ext)

    def _get_extension_data(self, name, ext=None):
        """Returns the data from an extension."""

        model = self.datamodel[name]
        ext_name = self._get_ext_name(model, ext)
        if not ext_name:
            return None

        if ext_name in self._extension_data:
            return self._extension_data[ext_name]

        if self.data_origin == 'file':
            ext_data = self.data[model.fits_extension(ext)].data

        elif self.data_origin == 'db':
            # If the table is "spaxel", this must be a 3D cube. If it is "cube",
            # uses self.data, which is basically the DataModelClass.Cube instance.
            if model.db_table == 'spaxel':
                ext_data = self.data.get3DCube(model.db_column(ext))
            elif model.db_table == 'cube':
                ext_data = getattr(self.data, model.db_column(ext))
            else:
                raise NotImplementedError('invalid db_table={!r}'.format(model.db_table))

        elif self.data_origin == 'api':

            params = {'release': self._release}
            url = marvin.config.urlmap['api']['getExtension']['url']

            try:
                response = self._toolInteraction(
                    url.format(name=self.plateifu,
                               cube_extension=model.fits_extension(ext).lower()),
                    params=params)
            except Exception as ee:
                raise MarvinError('found a problem when checking if remote cube '
                                  'exists: {0}'.format(str(ee)))

            data = response.getData()
            cube_ext_data = data['extension_data']
            ext_data = np.array(cube_ext_data) if cube_ext_data is not None else None

        self._extension_data[ext_name] = ext_data

        return ext_data

    def _get_spaxel_quantities(self, x, y):
        """Returns a dictionary of spaxel quantities."""

        cube_quantities = FuzzyDict({})

        if self.data_origin == 'db':

            session = marvin.marvindb.session
            datadb = marvin.marvindb.datadb

        if self.data_origin == 'file' or self.data_origin == 'db':

            # Stores a dictionary of (table, row)
            _db_rows = {}

            for dm in self.datamodel.datacubes + self.datamodel.spectra:

                data = {'value': None, 'ivar': None, 'mask': None, 'std': None}

                for key in data:

                    if key == 'ivar':
                        if dm in self.datamodel.spectra or not dm.has_ivar():
                            continue
                    if key == 'mask':
                        if dm in self.datamodel.spectra or not dm.has_mask():
                            continue
                    if key == 'std':
                        if dm in self.datamodel.datacubes or not dm.has_std():
                            continue

                    if self.data_origin == 'file':

                        extname = dm.fits_extension(None if key == 'value' else key)

                        if dm in self.datamodel.datacubes:
                            data[key] = self.data[extname].data[:, y, x]
                        else:
                            data[key] = self.data[extname].data

                    elif self.data_origin == 'db':

                        colname = dm.db_column(None if key == 'value' else key)

                        if dm in self.datamodel.datacubes:

                            if 'datacubes' not in _db_rows:
                                _db_rows['datacubes'] = session.query(datadb.Spaxel).filter(
                                    datadb.Spaxel.cube == self.data,
                                    datadb.Spaxel.x == x, datadb.Spaxel.y == y).one()

                            spaxel_data = getattr(_db_rows['datacubes'], colname)

                        else:

                            if 'spectra' not in _db_rows:
                                _db_rows['spectra'] = session.query(datadb.Cube).filter(
                                    datadb.Cube.pk == self.data.pk).one()

                            spaxel_data = getattr(_db_rows['spectra'], colname, None)

                        # In case the column was empty in the DB. At some point
                        # this can be removed.
                        if spaxel_data is None:
                            warnings.warn('cannot find {!r} data for {!r}. '
                                          'Maybe the data is not in the DB.'.format(
                                              colname, self.plateifu), MarvinUserWarning)
                            cube_quantities[dm.name] = None
                            continue

                        data[key] = np.array(spaxel_data)

                cube_quantities[dm.name] = Spectrum(data['value'],
                                                    ivar=data['ivar'],
                                                    mask=data['mask'],
                                                    std=data['std'],
                                                    wavelength=self._wavelength,
                                                    unit=dm.unit)

        if self.data_origin == 'api':

            params = {'release': self._release}
            url = marvin.config.urlmap['api']['getCubeQuantitiesSpaxel']['url']

            try:
                response = self._toolInteraction(url.format(name=self.plateifu,
                                                            x=x, y=y, params=params))
            except Exception as ee:
                raise MarvinError('found a problem when checking if remote cube '
                                  'exists: {0}'.format(str(ee)))

            data = response.getData()

            for dm in self.datamodel.datacubes + self.datamodel.spectra:

                if data[dm.name]['value'] is None:
                    warnings.warn('cannot find {!r} data for {!r}. '
                                  'Maybe the data is not in the DB.'.format(
                                      dm.name, self.plateifu), MarvinUserWarning)
                    cube_quantities[dm.name] = None
                    continue

                cube_quantities[dm.name] = Spectrum(data[dm.name]['value'],
                                                    ivar=data[dm.name]['ivar'],
                                                    mask=data[dm.name]['mask'],
                                                    wavelength=data['wavelength'],
                                                    unit=dm.unit)

        return cube_quantities

    @property
    def manga_target1(self):
        """Return MANGA_TARGET1 flag."""
        return get_manga_target('1', self._bitmasks, self.header)

    @property
    def manga_target2(self):
        """Return MANGA_TARGET2 flag."""
        return get_manga_target('2', self._bitmasks, self.header)

    @property
    def manga_target3(self):
        """Return MANGA_TARGET3 flag."""
        return get_manga_target('3', self._bitmasks, self.header)

    @property
    def target_flags(self):
        """Bundle MaNGA targeting flags."""
        return [self.manga_target1, self.manga_target2, self.manga_target3]

    @property
    def quality_flag(self):
        """Return ModelCube DAPQUAL flag."""
        drp3qual = self._bitmasks['MANGA_DRP3QUAL']
        drp3qual.mask = int(self.header['DRP3QUAL'])
        return drp3qual

    @property
    def pixmask(self):
        """Return the DRP3PIXMASK flag."""
        pixmask = self._bitmasks['MANGA_DRP3PIXMASK']
        pixmask.mask = getattr(self.flux, 'mask', None)
        return pixmask

    def getSpaxel(self, x=None, y=None, ra=None, dec=None,
                  properties=True, models=False, **kwargs):
        """Returns the :class:`~marvin.tools.spaxel.Spaxel` matching certain coordinates.

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
            models (`~marvin.tools.modelcube.ModelCube` or None or bool):
                A :class:`~marvin.tools.modelcube.ModelCube` object
                representing the DAP modelcube entity. If None, the |spaxel|
                will be returned without model information. Default is False.

        Returns:
            spaxels (list):
                The |spaxel|_ objects for this cube corresponding to the input
                coordinates. The length of the list is equal to the number
                of input coordinates.

        .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

        """

        return marvin.utils.general.general.getSpaxel(x=x, y=y, ra=ra, dec=dec,
                                                      cube=self,
                                                      maps=properties,
                                                      modelcube=models, **kwargs)

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

        raise NotImplementedError('getAperture is not currently implemented.')

    #     assert return_type in ['mask', 'mean', 'median', 'sum', 'spaxels']
    #
    #     if return_type not in ['mask', 'spaxels']:
    #         raise marvin.core.exceptions.MarvinNotImplemented(
    #             'return_type={0} is not yet implemented'.format(return_type))
    #
    #     if not photutils:
    #         raise MarvinError('getAperture currently requires photutils.')
    #
    #     if mode != 'pix':
    #         raise marvin.core.exceptions.MarvinNotImplemented(
    #             'mode={0} is not yet implemented'.format(mode))
    #
    #     if not np.isscalar(radius):
    #         raise marvin.core.exceptions.MarvinNotImplemented(
    #             'elliptical apertures are not yet implemented')
    #
    #     data_mask = np.zeros(self.shape)
    #
    #     if weight:
    #         phot_mode = ''
    #     else:
    #         phot_mode = 'center'
    #
    #     coords = np.atleast_2d(coords) + 0.5
    #
    #     mask = photutils.aperture_funcs.get_circular_fractions(
    #         data_mask, coords, radius, phot_mode, 0)
    #
    #     if return_type == 'mask':
    #         return mask
    #
    #     if return_type == 'spaxels':
    #         mask_idx = np.where(mask)
    #         spaxels = self.getSpaxel(x=mask_idx[0], y=mask_idx[1],
    #                                  xyorig='lower')
    #
    #         fractions = mask[mask_idx]
    #
    #         return (fractions, spaxels)
