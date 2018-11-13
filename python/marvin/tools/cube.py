#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-10-5
# @Filename: cube.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-12 18:40:51


from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import marvin
import marvin.core.exceptions
import marvin.tools.maps
import marvin.tools.spaxel
import marvin.utils.general.general
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.tools.quantities import DataCube, Spectrum
from marvin.utils.datamodel.drp import datamodel
from marvin.utils.general import FuzzyDict, get_nsa_data

from .core import MarvinToolsClass
from .mixins import GetApertureMixIn, NSAMixIn


class Cube(MarvinToolsClass, NSAMixIn, GetApertureMixIn):
    """A class to interface with MaNGA DRP data cubes.

    This class represents a fully reduced DRP data cube, initialised either
    from a file, a database, or remotely via the Marvin API.

    See `~.MarvinToolsClass` and `~.NSAMixIn` for a list of input parameters.

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

        self._init_attributes(self)

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

    @staticmethod
    def _init_attributes(obj):
        """Initialises several attributes."""

        obj.ra = float(obj.header['OBJRA'])
        obj.dec = float(obj.header['OBJDEC'])

        obj.mangaid = obj.header['MANGAID']

        obj._isbright = 'APOGEE' in obj.header['SRVYMODE']

        obj.dir3d = 'mastar' if obj._isbright else 'stack'

    @staticmethod
    def _do_file_checks(obj):
        """Performs a series of check when we load from a file."""

        obj.plateifu = obj.header['PLATEIFU']

        # Checks and populates the release.
        file_drpver = obj.header['VERSDRP3']
        file_drpver = 'v1_5_1' if file_drpver == 'v1_5_0' else file_drpver

        file_ver = marvin.config.lookUpRelease(file_drpver)
        assert file_ver is not None, 'cannot find file version.'

        if file_drpver != obj._drpver:
            warnings.warn('mismatch between file release={0} and object release={1}. '
                          'Setting object release to {0}'.format(file_ver, obj._release),
                          MarvinUserWarning)
            obj._release = file_ver

            # Reload NSA data from file version of drpall file
            obj._drpver, obj._dapver = marvin.config.lookUpVersions(release=obj._release)
            obj._drpall = marvin.config._getDrpAllPath(file_drpver)
            obj.mangaid = obj.header['MANGAID']

            nsa_source = 'drpall' if obj.nsa_source == 'auto' else obj.nsa_source

            obj._nsa = None
            obj._nsa = get_nsa_data(obj.mangaid, mode='auto', source=nsa_source,
                                    drpver=obj._drpver, drpall=obj._drpall)

        obj._drpver, obj._dapver = marvin.config.lookUpVersions(release=obj._release)

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

        self._check_file(self.data[0].header, self.data, 'Cube')

        self._wavelength = self.data['WAVE'].data
        self._shape = (self.data['FLUX'].header['NAXIS2'],
                       self.data['FLUX'].header['NAXIS1'])

        self._do_file_checks(self)

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
                            unit=model.unit, pixmask_flag=model.pixmask_flag)

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
                            unit=model.unit,
                            pixmask_flag=model.pixmask_flag)

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

    def _get_spaxel_quantities(self, x, y, spaxel=None):
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
                                                    unit=dm.unit,
                                                    pixmask_flag=dm.pixmask_flag)

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
                                                    unit=dm.unit,
                                                    pixmask_flag=dm.pixmask_flag)

        return cube_quantities

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

    def getRSS(self):
        """Returns the `~marvin.tools.rss.RSS` associated with this Cube."""

        return marvin.tools.RSS(plateifu=self.plateifu, mode=self.mode,
                                release=self.release)

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
