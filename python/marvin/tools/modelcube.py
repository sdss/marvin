#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-11-01
# @Filename: modelcube.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-12 18:41:23


from __future__ import absolute_import, division, print_function

import distutils
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import marvin
import marvin.core.exceptions
import marvin.tools.maps
import marvin.tools.spaxel
import marvin.utils.general.general
from marvin.core.exceptions import MarvinError
from marvin.tools.quantities import DataCube, Map, Spectrum
from marvin.utils.datamodel.dap import Model, datamodel
from marvin.utils.general import FuzzyDict

from .core import MarvinToolsClass
from .mixins import DAPallMixIn, GetApertureMixIn, NSAMixIn


class ModelCube(MarvinToolsClass, NSAMixIn, DAPallMixIn, GetApertureMixIn):
    """A class to interface with MaNGA DAP model cubes.

    This class represents a DAP model cube, initialised either from a file,
    a database, or remotely via the Marvin API. In addition to
    the parameters and variables defined for `~.MarvinToolsClass`, the
    following parameters and attributes are specific to `.Maps`.

    Parameters:
        bintype (str or None):
            The binning type. For MPL-4, one of the following: ``'NONE',
            'RADIAL', 'STON'`` (if ``None`` defaults to ``'NONE'``).
            For MPL-5, one of, ``'ALL', 'NRE', 'SPX', 'VOR10'``
            (defaults to ``'SPX'``). MPL-6 also accepts the ``'HYB10'`` binning
            schema.
        template (str or None):
            The stellar template used. For MPL-4, one of
            ``'M11-STELIB-ZSOL', 'MILES-THIN', 'MIUSCAT-THIN'`` (if ``None``,
            defaults to ``'MIUSCAT-THIN'``). For MPL-5 and successive, the only
            option in ``'GAU-MILESHC'`` (``None`` defaults to it).

    Attributes:
        header (`astropy.io.fits.Header`):
            The header of the datacube.
        wcs (`astropy.wcs.WCS`):
            The WCS solution for this plate

    """

    def __init__(self, input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None,
                 drpall=None, download=None, nsa_source='auto',
                 bintype=None, template=None, template_kin=None):

        if template_kin is not None:
            warnings.warn('template_kin is deprecated and will be removed in a future version.',
                          DeprecationWarning)
            template = template_kin if template is None else template

        # _set_datamodel will replace these strings with datamodel objects.
        self.bintype = bintype
        self.template = template
        self.datamodel = None

        self._bitmasks = None

        MarvinToolsClass.__init__(self, input=input, filename=filename,
                                  mangaid=mangaid, plateifu=plateifu,
                                  mode=mode, data=data, release=release,
                                  drpall=drpall, download=download)

        NSAMixIn.__init__(self, nsa_source=nsa_source)

        # Checks that DAP is at least MPL-5
        MPL5 = distutils.version.StrictVersion('2.0.2')
        if self.filename is None and distutils.version.StrictVersion(self._dapver) < MPL5:
            raise MarvinError('ModelCube requires at least dapver=\'2.0.2\'')

        self.header = None
        self.wcs = None
        self._wavelength = None
        self._redcorr = None
        self._shape = None

        # Model extensions
        self._extension_data = {}
        self._binned_flux = None
        self._redcorr = None
        self._full_fit = None
        self._emline_fit = None
        self._stellarcont_fit = None

        if self.data_origin == 'file':
            self._load_modelcube_from_file()
        elif self.data_origin == 'db':
            self._load_modelcube_from_db()
        elif self.data_origin == 'api':
            self._load_modelcube_from_api()
        else:
            raise marvin.core.exceptions.MarvinError(
                'data_origin={0} is not valid'.format(self.data_origin))

        # Confirm that drpver and dapver match the ones from the header.
        marvin.tools.maps.Maps._check_versions(self)

    def __repr__(self):
        """Representation for ModelCube."""

        return ('<Marvin ModelCube (plateifu={0!r}, mode={1!r}, data_origin={2!r}, '
                'bintype={3!r}, template={4!r})>'.format(self.plateifu,
                                                         self.mode,
                                                         self.data_origin,
                                                         str(self.bintype),
                                                         str(self.template)))

    def __getitem__(self, xy):
        """Returns the spaxel for ``(x, y)``"""

        return self.getSpaxel(x=xy[1], y=xy[0], xyorig='lower')

    def _set_datamodel(self):
        """Sets the datamodel, template, and bintype."""

        self.datamodel = datamodel[self.release].models
        self._bitmasks = datamodel[self.release].bitmasks
        self.bintype = self.datamodel.parent.get_bintype(self.bintype)
        self.template = self.datamodel.parent.get_template(self.template)

    def _getFullPath(self):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')
        daptype = '{0}-{1}'.format(self.bintype, self.template)

        return super(ModelCube, self)._getFullPath('mangadap5', ifu=ifu,
                                                   drpver=self._drpver,
                                                   dapver=self._dapver,
                                                   plate=plate, mode='LOGCUBE',
                                                   daptype=daptype)

    def download(self):
        """Downloads the cube using sdss_access - Rsync"""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')
        daptype = '{0}-{1}'.format(self.bintype, self.template)

        return super(ModelCube, self).download('mangadap5', ifu=ifu,
                                               drpver=self._drpver,
                                               dapver=self._dapver,
                                               plate=plate, mode='LOGCUBE',
                                               daptype=daptype)

    def _load_modelcube_from_file(self):
        """Initialises a model cube from a file."""

        if self.data is not None:
            assert isinstance(self.data, fits.HDUList), 'data is not an HDUList object'
        else:
            try:
                self.data = fits.open(self.filename)
            except IOError as err:
                raise IOError('filename {0} cannot be found: {1}'.format(self.filename, err))

        self.header = self.data[0].header
        self._check_file(self.header, self.data, 'ModelCube')
        self.wcs = WCS(self.data['FLUX'].header)
        self._wavelength = self.data['WAVE'].data
        self._redcorr = self.data['REDCORR'].data
        self._shape = (self.data['FLUX'].header['NAXIS2'],
                       self.data['FLUX'].header['NAXIS1'])

        self.plateifu = self.header['PLATEIFU']
        self.mangaid = self.header['MANGAID']

        # Checks and populates release.
        file_drpver = self.header['VERSDRP3']
        file_drpver = 'v1_5_1' if file_drpver == 'v1_5_0' else file_drpver

        file_ver = marvin.config.lookUpRelease(file_drpver)
        assert file_ver is not None, 'cannot find file version.'

        if file_drpver != self._drpver:
            warnings.warn('mismatch between file version={0} and object release={1}. '
                          'Setting object release to {0}'.format(file_ver, self._release),
                          marvin.core.exceptions.MarvinUserWarning)
            self._release = file_ver

        self._drpver, self._dapver = marvin.config.lookUpVersions(release=self._release)

        # Updates datamodel, bintype, and template with the versions from the header.
        self.datamodel = datamodel[self._dapver].models
        self.bintype = self.datamodel.parent.get_bintype(self.header['BINKEY'].strip().upper())
        self.template = self.datamodel.parent.get_template(self.header['SCKEY'].strip().upper())

    def _load_modelcube_from_db(self):
        """Initialises a model cube from the DB."""

        mdb = marvin.marvindb
        plate, ifu = self.plateifu.split('-')

        if not mdb.isdbconnected:
            raise MarvinError('No db connected')

        else:

            datadb = mdb.datadb
            dapdb = mdb.dapdb

            dm = datamodel[self.release]
            if dm.db_only:
                if self.bintype not in dm.db_only:
                    raise marvin.core.exceptions.MarvinError(
                        'Specified bintype {0} is not '
                        'available in the DB'.format(self.bintype.name))

            if self.data:
                assert isinstance(self.data, dapdb.ModelCube), \
                    'data is not an instance of marvindb.dapdb.ModelCube.'
            else:
                # Initial query for version
                version_query = mdb.session.query(dapdb.ModelCube).join(
                    dapdb.File,
                    datadb.PipelineInfo,
                    datadb.PipelineVersion).filter(
                        datadb.PipelineVersion.version == self._dapver).from_self()

                # Query for model cube parameters
                db_modelcube = version_query.join(
                    dapdb.File,
                    datadb.Cube,
                    datadb.IFUDesign).filter(
                        datadb.Cube.plate == plate,
                        datadb.IFUDesign.name == str(ifu)).from_self().join(
                            dapdb.File,
                            dapdb.FileType).filter(dapdb.FileType.value == 'LOGCUBE').join(
                                dapdb.Structure, dapdb.BinType).join(
                                    dapdb.Template,
                                    dapdb.Structure.template_kin_pk == dapdb.Template.pk).filter(
                                        dapdb.BinType.name == self.bintype.name,
                                        dapdb.Template.name == self.template.name).all()

                if len(db_modelcube) > 1:
                    raise MarvinError('more than one ModelCube found for '
                                      'this combination of parameters.')

                elif len(db_modelcube) == 0:
                    raise MarvinError('no ModelCube found for this combination of parameters.')

                self.data = db_modelcube[0]

            self.header = self.data.file.primary_header
            self.wcs = WCS(self.data.file.cube.wcs.makeHeader())
            self._wavelength = np.array(self.data.file.cube.wavelength.wavelength, dtype=np.float)
            self._redcorr = np.array(self.data.redcorr[0].value, dtype=np.float)
            self._shape = self.data.file.cube.shape.shape

            self.plateifu = str(self.header['PLATEIFU'].strip())
            self.mangaid = str(self.header['MANGAID'].strip())

    def _load_modelcube_from_api(self):
        """Initialises a model cube from the API."""

        url = marvin.config.urlmap['api']['getModelCube']['url']
        url_full = url.format(name=self.plateifu, bintype=self.bintype.name,
                              template=self.template.name)

        try:
            response = self._toolInteraction(url_full)
        except Exception as ee:
            raise MarvinError('found a problem when checking if remote model cube '
                              'exists: {0}'.format(str(ee)))

        data = response.getData()

        self.header = fits.Header.fromstring(data['header'])
        self.wcs = WCS(fits.Header.fromstring(data['wcs_header']))
        self._wavelength = np.array(data['wavelength'])
        self._redcorr = np.array(data['redcorr'])
        self._shape = tuple(data['shape'])

        self.plateifu = str(self.header['PLATEIFU'].strip())
        self.mangaid = str(self.header['MANGAID'].strip())

    def getSpaxel(self, x=None, y=None, ra=None, dec=None,
                  drp=True, properties=True, **kwargs):
        """Returns the :class:`~marvin.tools.spaxel.Spaxel` matching certain coordinates.

        The coordinates of the spaxel to return can be input as ``x, y`` pixels
        relative to``xyorig`` in the cube, or as ``ra, dec`` celestial
        coordinates.

        If ``spectrum=True``, the returned |spaxel| will be instantiated with the
        DRP spectrum of the spaxel for the DRP cube associated with this
        ModelCube. The same is true for ``properties=True`` for the DAP
        properties of the spaxel in the Maps associated with these coordinates.

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
            drpa (bool):
                If ``True``, the |spaxel| will be initialised with the
                corresponding DRP data.
            properties (bool):
                If ``True``, the |spaxel| will be initialised with the
                corresponding DAP properties for this spaxel.

        Returns:
            spaxels (list):
                The |spaxel|_ objects for this cube/maps corresponding to the
                input coordinates. The length of the list is equal to the
                number of input coordinates.

        .. |spaxel| replace:: :class:`~marvin.tools.spaxel.Spaxel`

        """

        return marvin.utils.general.general.getSpaxel(
            x=x, y=y, ra=ra, dec=dec,
            cube=drp, maps=properties, modelcube=self, **kwargs)

    def _get_extension_data(self, name, ext=None):
        """Returns the data from an extension."""

        model = self.datamodel[name]
        ext_name = model.fits_extension(ext)

        if ext_name in self._extension_data:
            return self._extension_data[ext_name]

        if self.data_origin == 'file':
            ext_data = self.data[model.fits_extension(ext)].data

        elif self.data_origin == 'db':
            # If the table is "spaxel", this must be a 3D cube. If it is "cube",
            # uses self.data, which is basically the DataModelClass.Cube instance.
            ext_data = self.data.get3DCube(model.db_column(ext))

        elif self.data_origin == 'api':

            params = {'release': self._release}
            url = marvin.config.urlmap['api']['getModelCubeExtension']['url']

            try:
                response = self._toolInteraction(
                    url.format(name=self.plateifu,
                               modelcube_extension=model.fits_extension(ext).lower(),
                               bintype=self.bintype.name, template=self.template.name),
                    params=params)
            except Exception as ee:
                raise MarvinError('found a problem when checking if remote '
                                  'modelcube exists: {0}'.format(str(ee)))

            data = response.getData()
            cube_ext_data = data['extension_data']
            ext_data = np.array(cube_ext_data) if cube_ext_data is not None else None

        self._extension_data[ext_name] = ext_data

        return ext_data

    def _get_spaxel_quantities(self, x, y, spaxel=None):
        """Returns a dictionary of spaxel quantities."""

        modelcube_quantities = FuzzyDict({})

        if self.data_origin == 'db':

            session = marvin.marvindb.session
            dapdb = marvin.marvindb.dapdb

        if self.data_origin == 'file' or self.data_origin == 'db':

            _db_row = None

            for dm in self.datamodel:

                data = {'value': None, 'ivar': None, 'mask': None}

                for key in data:

                    if key == 'ivar' and not dm.has_ivar():
                        continue
                    if key == 'mask' and not dm.has_mask():
                        continue

                    if self.data_origin == 'file':

                        extname = dm.fits_extension(None if key == 'value' else key)
                        data[key] = self.data[extname].data[:, y, x]

                    elif self.data_origin == 'db':

                        colname = dm.db_column(None if key == 'value' else key)

                        if not _db_row:
                            _db_row = session.query(dapdb.ModelSpaxel).filter(
                                dapdb.ModelSpaxel.modelcube_pk == self.data.pk,
                                dapdb.ModelSpaxel.x == x, dapdb.ModelSpaxel.y == y).one()

                        data[key] = np.array(getattr(_db_row, colname))

                quantity = Spectrum(data['value'], ivar=data['ivar'], mask=data['mask'],
                                    wavelength=self._wavelength, unit=dm.unit,
                                    pixmask_flag=dm.pixmask_flag)

                if spaxel:
                    quantity._init_bin(spaxel=spaxel, parent=self, datamodel=dm)

                modelcube_quantities[dm.full()] = quantity

        if self.data_origin == 'api':

            params = {'release': self._release}
            url = marvin.config.urlmap['api']['getModelCubeQuantitiesSpaxel']['url']

            try:
                response = self._toolInteraction(url.format(name=self.plateifu,
                                                            x=x, y=y,
                                                            bintype=self.bintype.name,
                                                            template=self.template.name,
                                                            params=params))
            except Exception as ee:
                raise MarvinError('found a problem when checking if remote modelcube '
                                  'exists: {0}'.format(str(ee)))

            data = response.getData()

            for dm in self.datamodel:

                quantity = Spectrum(data[dm.name]['value'], ivar=data[dm.name]['ivar'],
                                    mask=data[dm.name]['mask'], wavelength=data['wavelength'],
                                    unit=dm.unit, pixmask_flag=dm.pixmask_flag)

                if spaxel:
                    quantity._init_bin(spaxel=spaxel, parent=self, datamodel=dm)

                modelcube_quantities[dm.full()] = quantity

        return modelcube_quantities

    def get_binid(self, model=None):
        """Returns the binid map associated with a model.

        Parameters
        ----------
        model : `datamodel.Model` or None
            The model for which the associated binid map will be returned.
            If ``binid=None``, the default binid is returned.

        Returns
        -------
        binid : `Map`
            A `Map` with the binid associated with ``model`` or the default
            binid.

        """

        assert model is None or isinstance(model, Model), 'invalid model type.'

        if model is not None:
            binid_prop = model.binid
        else:
            binid_prop = self.datamodel.parent.default_binid

        # Before MPL-6, the modelcube does not include the binid extension,
        # so we need to get the binid map from the associated MAPS.
        if (distutils.version.StrictVersion(self._dapver) <
                distutils.version.StrictVersion('2.1')):
            return self.getMaps().get_binid()

        if self.data_origin == 'file':

            if binid_prop.channel is None:
                binid_map_data = self.data[binid_prop.name].data[:, :]
            else:
                binid_map_data = self.data[binid_prop.name].data[binid_prop.channel.idx, :, :]

        elif self.data_origin == 'db':

            mdb = marvin.marvindb

            table = mdb.dapdb.ModelSpaxel
            column = getattr(table, binid_prop.db_column())

            binid_list = mdb.session.query(column).filter(
                table.modelcube_pk == self.data.pk).order_by(table.x, table.y).all()

            nx = ny = int(np.sqrt(len(binid_list)))
            binid_array = np.array(binid_list)

            binid_map_data = binid_array.transpose().reshape((ny, nx)).transpose(1, 0)

        elif self.data_origin == 'api':

            params = {'release': self._release}
            url = marvin.config.urlmap['api']['getModelCubeBinid']['url']

            extension = model.fits_extension().lower() if model is not None else 'flux'

            try:
                response = self._toolInteraction(
                    url.format(name=self.plateifu,
                               modelcube_extension=extension,
                               bintype=self.bintype.name,
                               template=self.template.name), params=params)
            except Exception as ee:
                raise MarvinError('found a problem when checking if remote '
                                  'modelcube exists: {0}'.format(str(ee)))

            if response.results['error'] is not None:
                raise MarvinError('found a problem while getting the binid from API: {}'
                                  .format(str(response.results['error'])))

            binid_map_data = np.array(response.getData()['binid'])

        binid_map = Map(binid_map_data, unit=binid_prop.unit)
        binid_map._datamodel = binid_prop

        return binid_map

    @property
    def binned_flux(self):
        """Returns the binned flux datacube."""

        model = self.datamodel['binned_flux']

        binned_flux_array = self._get_extension_data('flux')
        binned_flux_ivar = self._get_extension_data('flux', 'ivar')
        binned_flux_mask = self._get_extension_data('flux', 'mask')

        return DataCube(binned_flux_array,
                        np.array(self._wavelength),
                        ivar=binned_flux_ivar,
                        mask=binned_flux_mask,
                        redcorr=self._redcorr,
                        binid=self.get_binid(model),
                        unit=model.unit,
                        pixmask_flag=model.pixmask_flag)

    @property
    def full_fit(self):
        """Returns the full fit datacube."""

        model = self.datamodel['full_fit']

        model_array = self._get_extension_data('full_fit')
        model_mask = self._get_extension_data('flux', 'mask')

        return DataCube(model_array,
                        np.array(self._wavelength),
                        ivar=None,
                        mask=model_mask,
                        redcorr=self._redcorr,
                        binid=self.get_binid(model),
                        unit=model.unit,
                        pixmask_flag=model.pixmask_flag)

    @property
    def emline_fit(self):
        """Returns the emission line fit."""

        model = self.datamodel['emline_fit']

        emline_array = self._get_extension_data('emline_fit')
        emline_mask = self._get_extension_data('emline_fit', 'mask')

        return DataCube(emline_array,
                        np.array(self._wavelength),
                        ivar=None,
                        mask=emline_mask,
                        redcorr=self._redcorr,
                        binid=self.get_binid(model),
                        unit=model.unit,
                        pixmask_flag=model.pixmask_flag)

    @property
    def stellarcont_fit(self):
        """Returns the stellar continuum fit."""

        array = (self._get_extension_data('full_fit') -
                 self._get_extension_data('emline_fit') -
                 self._get_extension_data('emline_base_fit'))

        model = self.datamodel['full_fit']

        stellarcont_mask = self._get_extension_data('flux', 'mask')

        return DataCube(array,
                        np.array(self._wavelength),
                        ivar=None,
                        mask=stellarcont_mask,
                        redcorr=self._redcorr,
                        binid=self.get_binid(model),
                        unit=model.unit,
                        pixmask_flag=model.pixmask_flag)

    def getCube(self):
        """Returns the associated `~marvin.tools.cube.Cube`."""

        if self.data_origin == 'db':
            cube_data = self.data.file.cube
        else:
            cube_data = None

        return marvin.tools.cube.Cube(data=cube_data,
                                      plateifu=self.plateifu,
                                      release=self.release)

    def getMaps(self):
        """Returns the associated`~marvin.tools.maps.Maps`."""

        return marvin.tools.maps.Maps(plateifu=self.plateifu,
                                      bintype=self.bintype,
                                      template=self.template,
                                      release=self.release)

    def is_binned(self):
        """Returns True if the ModelCube is not unbinned."""

        return self.bintype.binned

    def get_unbinned(self):
        """Returns a version of ``self`` corresponding to the unbinned ModelCube."""

        if not self.is_binned:
            return self
        else:
            return ModelCube(plateifu=self.plateifu, release=self.release,
                             bintype=self.datamodel.parent.get_unbinned(),
                             template=self.template,
                             mode=self.mode)
