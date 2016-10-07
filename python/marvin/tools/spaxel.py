#!/usr/bin/env python
# encoding: utf-8
#
# spaxel.py
#
# Licensed under a 3-clause BSD license.
#
# Revision history:
#     11 Apr 2016 J. Sánchez-Gallego
#       Initial version


from __future__ import division
from __future__ import print_function

import warnings

import numpy as np

import marvin
import marvin.core.core
import marvin.core.exceptions
import marvin.utils.general.general
import marvin.tools.cube
import marvin.tools.maps
import marvin.tools.modelcube

from marvin.api import api
from marvin.core import MarvinError, MarvinUserWarning
from marvin.tools.analysis_props import AnalysisProperty, DictOfProperties
from marvin.tools.spectrum import Spectrum


class Spaxel(object):
    """A class to interface with a spaxel in a cube.

    This class represents an spaxel with information from the reduced DRP
    spectrum, the DAP maps properties, and the model spectrum from the DAP
    logcube. A ``Spaxel`` can be initialised with all or only part of that
    information, and either from a file, a database, or remotely via the
    Marvin API.

    Parameters:
        x,y (int):
            The `x` and `y` coordinates of the spaxel in the cube (0-indexed).
        cube_filename (str):
            The path of the data cube file containing the spaxel to load.
        maps_filename (str):
            The path of the DAP Maps file containing the spaxel to load.
        modelcube_filename (str):
            The path of the DAP model cube file containing the spaxel to load.
        mangaid (str):
            The mangaid of the cube/maps of the spaxel to load.
        plateifu (str):
            The plate-ifu of the cube/maps of the spaxel to load (either
            ``mangaid`` or ``plateifu`` can be used, but not both).
        cube (:class:`~marvin.tools.cube.Cube` object or bool):
            If ``cube`` is a :class:`~marvin.tools.cube.Cube` object, that
            cube will be used for the ``Spaxel`` initilisitation. This mode
            is mostly intended for
            :class:`~marvin.utils.general.general.getSpaxel` as it
            significantly improves loading time. Otherwise, ``cube`` can be
            ``True`` (default), in which case a cube will be instantiated using
            the input ``filename``, ``mangaid``, or ``plateifu``. If
            ``cube=False``, no cube will be used and ``Spaxel.spectrum`` will
            not be populated.
        maps (:class:`~marvin.tools.maps.Maps` object or bool)
            As ``cube`` but populates ``Spaxel.properties`` with a dictionary
            of DAP measurements corresponding to the spaxel in the maps that
            matches ``bintype``, ``template_kin``, and ``template_pop``.
        modelcube (:class:`marvin.tools.modelcube.ModelCube` object or bool)
            As ``cube`` but populates ``Spaxel.model_flux``, ``Spaxel.model``,
            ``Spaxel.redcorr``, ``Spaxel.emline``, ``Spaxel.emline_base``, and
            ``Spaxel.stellar_continuum`` from the corresponding
            spaxel of the DAP modelcube that matches ``bintype``,
            ``template_kin``, and ``template_pop``.
        bintype (str or None):
            The binning type. For MPL-4, one of the following: ``'NONE',
            'RADIAL', 'STON'`` (if ``None`` defaults to ``'NONE'``).
            For MPL-5 and successive, one of, ``'ALL', 'NRE', 'SPX', 'VOR10'``
            (defaults to ``'ALL'``).
        template_kin (str or None):
            The template use for kinematics. For MPL-4, one of
            ``'M11-STELIB-ZSOL', 'MILES-THIN', 'MIUSCAT-THIN'`` (if ``None``,
            defaults to ``'MIUSCAT-THIN'``). For MPL-5 and successive, the only
            option in ``'GAU-MILESHC'`` (``None`` defaults to it).
        template_pop (str or None):
            A placeholder for a future version in which stellar populations
            are fitted using a different template that ``template_kin``. It
            has no effect for now.
        mplver,drver (str):
            The MPL/DR version of the data to use. Only one ``mplver`` or
            ``drver`` must be defined at the same time.

    Attributes:
        spectrum (:class:`~marvin.tools.spectrum.Spectrum` object):
            A `Spectrum` object with the DRP spectrum and associated ivar and
            mask for this spaxel.
        specres (Numpy array):
            Median spectral resolution as a function of wavelength for the
            fibers in this IFU.
        specresd (Numpy array):
            Standard deviation of spectral resolution as a function of
            wavelength for the fibers in this IFU.
        properties (:class:`~marvin.tools.analysis_props.DictOfProperties`):
            A dotable, case-insensitive dictionary of
            :class:`~marvin.tools.analysis_props.AnalysisProperty` objects
            from the DAP maps extensions. The keys are a combination of
            category and channel, when applicable, e.g.,
            ``emline_sflux_siii_9533``.
        model_flux (:class:`~marvin.tools.spectrum.Spectrum` object):
            A `Spectrum` object with the flux of the binned spectrum. Includes
            ``ivar`` and ``mask``.
        wavelength (Numpy array):
            Wavelength vector, in Angstrom.
        redcorr (Numpy array):
            Reddening correction applied during the fitting procedures;
            ``dereddened_flux = model_flux.flux * redcorr``.
        model (:class:`~marvin.tools.spectrum.Spectrum` object):
            The best fitting model spectra (sum of the fitted continuum and
            emission-line models). Includes ``mask``.
        emline (:class:`~marvin.tools.spectrum.Spectrum` object):
            The model spectrum with only the emission lines. Includes ``mask``.
        emline_base (:class:`~marvin.tools.spectrum.Spectrum` object):
            The bitmask that only applies to the emission-line modeling.
            Includes ``mask``.

    """

    def __init__(self, *args, **kwargs):

        valid_kwargs = [
            'x', 'y', 'cube_filename', 'maps_filename', 'modelcube_filename',
            'mangaid', 'plateifu', 'cube', 'maps', 'modelcube', 'bintype',
            'template_kin', 'template_pop', 'mplver', 'drver']

        assert len(args) == 0, 'Spaxel does not accept arguments, only keywords.'
        for kw in kwargs:
            assert kw in valid_kwargs, 'keyword {0} is not valid'.format(kw)

        self.cube = kwargs.pop('cube', True) or False
        self.maps = kwargs.pop('maps', True) or False
        self.modelcube = kwargs.pop('modelcube', True) or False

        if not self.cube and not self.maps and not self.modelcube:
            raise MarvinError('either cube, maps, or modelcube must be True or '
                              'a Marvin Cube, Maps, or ModelCube object must be specified.')

        # Checks versions
        input_mplver = kwargs.pop('mplver', None)
        input_drver = kwargs.pop('drver', None)

        if not input_mplver and not input_drver:
            input_mplver = marvin.config.mplver
            input_drver = marvin.config.drver

        assert bool(input_mplver) != bool(input_drver), ('one and only one of '
                                                         'drver or mplver must be set.')

        self._mplver, self._drver = self._check_version(input_mplver, input_drver)

        self._drpver, self._dapver = marvin.config.lookUpVersions(mplver=self._mplver,
                                                                  drver=self._drver)

        self.plateifu = None
        self.mangaid = None

        if len(args) > 0:
            self.x = args[0]
            self.y = args[1]
        else:
            self.x = kwargs.pop('x', None)
            self.y = kwargs.pop('y', None)

        assert self.x is not None and self.y is not None, 'Spaxel requires x and y to initialise.'

        self.specres = None
        self.specresd = None
        self.spectrum = None
        self.properties = {}

        self.model_flux = None
        self.redcorr = None
        self.model = None
        self.emline = None
        self.emline_base = None
        self.stellar_continuum = None

        self.plateifu = kwargs.pop('plateifu', None)
        self.mangaid = kwargs.pop('mangaid', None)

        cube_filename = kwargs.pop('cube_filename', None)
        self._check_cube(cube_filename)

        self.bintype = None
        self.template_kin = None

        if self.maps or self.modelcube:

            # Some versions, like DR13, don't have an associated DAP, so we check.
            assert self._dapver, 'this MPL/DR version does not have an associated dapver.'

            self.bintype = marvin.tools.maps._get_bintype(
                self._dapver, bintype=kwargs.get('bintype', None))
            self.template_kin = marvin.tools.maps._get_template_kin(
                self._dapver, template_kin=kwargs.get('template_kin', None))

        maps_filename = kwargs.pop('maps_filename', None)
        self._check_maps(maps_filename)

        modelcube_filename = kwargs.pop('modelcube_filename', None)
        self._check_modelcube(modelcube_filename)

    def _check_version(self, mplver, drver):

        has_cube = isinstance(self.cube, marvin.tools.cube.Cube)
        has_maps = isinstance(self.maps, marvin.tools.maps.Maps)
        has_modelcube = isinstance(self.modelcube, marvin.tools.modelcube.ModelCube)

        if not has_cube and not has_maps and not has_modelcube:
            return mplver, drver

        if has_cube and has_maps:

            assert self.cube._mplver == self.maps._mplver
            assert self.cube._drver == self.maps._drver

            if has_modelcube:
                assert self.cube._mplver == self.modelcube._mplver
                assert self.cube._drver == self.modelcube._drver

            return self.cube._mplver, self.cube._drver

        if has_cube and has_modelcube:
            assert self.cube._mplver == self.modelcube._mplver
            assert self.cube._drver == self.modelcube._drver
            return self.cube._mplver, self.cube._drver

        if has_maps and has_modelcube:
            assert self.maps._mplver == self.modelcube._mplver
            assert self.maps._drver == self.modelcube._drver
            return self.maps._mplver, self.maps._drver

        if has_cube:
            return self.cube._mplver, self.cube._drver

        if has_maps:
            return self.maps._mplver, self.maps._drver

        if has_modelcube:
            return self.modelcube._mplver, self.modelcube._drver

    def _check_cube(self, cube_filename):
        """Loads the cube and the spectrum."""

        # Checks that the cube is correct or load ones if cube == True.
        if not isinstance(self.cube, bool):
            assert isinstance(self.cube, marvin.tools.cube.Cube), \
                'cube is not an instance of marvin.tools.cube.Cube or a boolean.'
        elif self.cube is True:
            self.cube = marvin.tools.cube.Cube(filename=cube_filename,
                                               plateifu=self.plateifu,
                                               mangaid=self.mangaid,
                                               mplver=self._mplver, drver=self._drver)
        else:
            self.cube = None
            return

        if self.plateifu is not None:
            assert self.plateifu == self.cube.plateifu, \
                'input plateifu does not match the cube plateifu. '
        else:
            self.plateifu = self.cube.plateifu

        if self.mangaid is not None:
            assert self.mangaid == self.cube.mangaid, \
                'input mangaid does not match the cube mangaid. '
        else:
            self.mangaid = self.cube.mangaid

        self._parent_shape = self.cube.shape

        # Loads the spectrum
        self._load_spectrum()

    def _check_maps(self, maps_filename):
        """Loads the cube and the properties."""

        if not isinstance(self.maps, bool):
            assert isinstance(self.maps, marvin.tools.maps.Maps), \
                'maps is not an instance of marvin.tools.maps.Maps or a boolean.'
        elif self.maps is True:
            self.maps = marvin.tools.maps.Maps(filename=maps_filename,
                                               mangaid=self.mangaid,
                                               plateifu=self.plateifu,
                                               bintype=self.bintype,
                                               template_kin=self.template_kin,
                                               mplver=self._mplver, drver=self._drver)
        else:
            self.maps = None
            return

        if self.plateifu is not None:
            assert self.plateifu == self.maps.plateifu, \
                'input plateifu does not match the maps plateifu. '
        else:
            self.plateifu = self.maps.plateifu

        if self.mangaid is not None:
            assert self.mangaid == self.maps.mangaid, \
                'input mangaid does not match the maps mangaid. '
        else:
            self.mangaid = self.maps.mangaid

        self._parent_shape = self.maps.shape

        self.bintype = self.maps.bintype
        self.template_kin = self.maps.template_kin

        # Loads the properties
        self._load_properties()

    def _check_modelcube(self, modelcube_filename):
        """Loads the modelcube and associated arrays."""

        if not isinstance(self.modelcube, bool):
            assert isinstance(self.modelcube, marvin.tools.modelcube.ModelCube), \
                'modelcube is not an instance of marvin.tools.modelcube.ModelCube or a boolean.'
        elif self.modelcube is True:

            if self._is_MPL4():
                warnings.warn('ModelCube cannot be instantiated for MPL-4.',
                              MarvinUserWarning)
                self.modelcube = None
                return

            self.modelcube = marvin.tools.modelcube.ModelCube(filename=modelcube_filename,
                                                              mangaid=self.mangaid,
                                                              plateifu=self.plateifu,
                                                              bintype=self.bintype,
                                                              template_kin=self.template_kin,
                                                              mplver=self._mplver,
                                                              drver=self._drver)
        else:
            self.modelcube = None
            return

        self.bintype = self.modelcube.bintype
        self.template_kin = self.modelcube.template_kin

        if self.plateifu is not None:
            assert self.plateifu == self.modelcube.plateifu, \
                'input plateifu does not match the modelcube plateifu. '
        else:
            self.plateifu = self.modelcube.plateifu

        if self.mangaid is not None:
            assert self.mangaid == self.modelcube.mangaid, \
                'input mangaid does not match the modelcube mangaid. '
        else:
            self.mangaid = self.modelcube.mangaid

        self._parent_shape = self.modelcube.shape

        self._load_models()

    def __repr__(self):
        """Spaxel representation."""

        # Gets the coordinates relative to the centre of the cube/maps.
        yMid, xMid = np.array(self._parent_shape) / 2.
        xCentre = int(self.x - xMid)
        yCentre = int(self.y - yMid)

        return ('<Marvin Spaxel (x={0.x:d}, y={0.y:d}; x_cen={1:d}, y_cen={2:d}>'.format(self,
                                                                                         xCentre,
                                                                                         yCentre))

    def _is_MPL4(self):
        """Returns True if the dapver correspond to MPL-4."""

        if self._dapver == '1.1.1':
            return True
        return False

    def _load_spectrum(self):
        """Initialises Spaxel.spectrum."""

        assert self.cube, 'a valid cube is needed to initialise the spectrum.'

        if self.cube.data_origin == 'file':

            cube_hdu = self.cube.data

            self.spectrum = Spectrum(cube_hdu['FLUX'].data[:, self.y, self.x],
                                     units='1E-17 erg/s/cm^2/Ang/spaxel',
                                     wavelength=cube_hdu['WAVE'].data,
                                     wavelength_unit='Angstrom',
                                     ivar=cube_hdu['IVAR'].data[:, self.y, self.x],
                                     mask=cube_hdu['MASK'].data[:, self.y, self.x])

            self.specres = cube_hdu['SPECRES'].data
            self.specresd = cube_hdu['SPECRESD'].data

        elif self.cube.data_origin == 'db':

            if marvin.marvindb is None:
                raise MarvinError('there is not a valid DB connection.')

            session = marvin.marvindb.session
            datadb = marvin.marvindb.datadb

            cube_db = self.cube.data

            spaxel = session.query(datadb.Spaxel).filter(
                datadb.Spaxel.cube == cube_db,
                datadb.Spaxel.x == self.x, datadb.Spaxel.y == self.y).one()

            if spaxel is None:
                raise MarvinError('cannot find an spaxel for x={0.x}, y={0.y}'.format(self))

            self.spectrum = Spectrum(spaxel.flux,
                                     units='1E-17 erg/s/cm^2/Ang/spaxel',
                                     wavelength=cube_db.wavelength.wavelength,
                                     wavelength_unit='Angstrom',
                                     ivar=spaxel.ivar,
                                     mask=spaxel.mask)

            self.specres = np.array(cube_db.specres)
            self.specresd = None

        elif self.cube.data_origin == 'api':

            # Calls the API to retrieve the DRP spectrum information for this spaxel.

            routeparams = {'name': self.plateifu, 'x': self.x, 'y': self.y}

            url = marvin.config.urlmap['api']['getSpectrum']['url'].format(**routeparams)

            # Make the API call
            response = api.Interaction(url, params={'mplver': self._mplver,
                                                    'drver': self._drver})

            # Temporarily stores the arrays prior to subclassing from np.array
            data = response.getData()

            # Instantiates the spectrum from the returned values from the Interaction
            self.spectrum = Spectrum(data['flux'],
                                     units='1E-17 erg/s/cm^2/Ang/spaxel',
                                     wavelength=data['wavelength'],
                                     wavelength_unit='Angstrom',
                                     ivar=data['ivar'],
                                     mask=data['mask'])

            self.specres = np.array(data['specres'])
            self.specresd = None

            return response

    def _load_properties(self):
        """Initialises Spaxel.properties."""

        assert self.maps, 'a valid maps is needed to initialise the properties.'

        maps_properties = self.maps.properties

        if self.maps.data_origin == 'file':

            maps_hdu = self.maps.data

            properties = {}
            for prop in maps_properties:

                prop_hdu = maps_hdu[prop.name]
                prop_hdu_ivar = None if not prop.ivar else maps_hdu[prop.name + '_ivar']
                prop_hdu_mask = None if not prop.mask else maps_hdu[prop.name + '_mask']

                if prop.channels:
                    for ii, channel in enumerate(prop.channels):

                        if isinstance(prop.unit, str) or not prop.unit:
                            unit = prop.unit
                        else:
                            unit = prop.unit[ii]

                        properties[prop.fullname(channel=channel)] = AnalysisProperty(
                            prop.name,
                            channel=channel,
                            value=prop_hdu.data[ii, self.y, self.x],
                            ivar=prop_hdu_ivar.data[ii, self.y, self.x] if prop_hdu_ivar else None,
                            mask=prop_hdu_mask.data[ii, self.y, self.x] if prop_hdu_mask else None,
                            unit=unit,
                            description=prop.description)

                else:

                    properties[prop.fullname(channel=channel)] = AnalysisProperty(
                        prop.name,
                        channel=channel,
                        value=prop_hdu.data[self.y, self.x],
                        ivar=prop_hdu_ivar.data[self.y, self.x] if prop_hdu_ivar else None,
                        mask=prop_hdu_mask.data[self.y, self.x] if prop_hdu_mask else None,
                        unit=unit,
                        description=prop.description)

        elif self.maps.data_origin == 'db':

            if marvin.marvindb is None:
                raise MarvinError('there is not a valid DB connection.')

            session = marvin.marvindb.session
            dapdb = marvin.marvindb.dapdb

            # Gets the spaxel_index for this spaxel.
            spaxel_index = self.x * self.maps.shape[0] + self.y

            spaxelprops_table = dapdb.SpaxelProp if self._is_MPL4() else dapdb.SpaxelProp5
            spaxelprops = session.query(spaxelprops_table).filter(
                spaxelprops_table.file == self.maps.data,
                spaxelprops_table.spaxel_index == spaxel_index).one()

            if spaxelprops is None:
                raise MarvinError('cannot find an spaxelprops for x={0.x}, y={0.y}'.format(self))

            properties = {}
            for prop in maps_properties:

                if prop.channels:

                    for ii, channel in enumerate(prop.channels):

                        if isinstance(prop.unit, str) or not prop.unit:
                            unit = prop.unit
                        else:
                            unit = prop.unit[ii]

                        properties[prop.fullname(channel=channel)] = AnalysisProperty(
                            prop.name,
                            channel=channel,
                            value=getattr(spaxelprops, prop.fullname(channel=channel)),
                            ivar=(getattr(spaxelprops, prop.fullname(channel=channel, ext='ivar'))
                                  if prop.ivar else None),
                            mask=(getattr(spaxelprops, prop.fullname(channel=channel, ext='mask'))
                                  if prop.mask else None),
                            unit=unit,
                            description=prop.description)

                else:

                    properties[prop.fullname()] = AnalysisProperty(
                        prop.name,
                        channel=None,
                        value=getattr(spaxelprops, prop.fullname()),
                        ivar=(getattr(spaxelprops, prop.fullname(ext='ivar'))
                              if prop.ivar else None),
                        mask=(getattr(spaxelprops, prop.fullname(ext='mask'))
                              if prop.mask else None),
                        unit=prop.unit,
                        description=prop.description)

        elif self.maps.data_origin == 'api':

            # Calls /api/<name>/properties/<path:path> to retrieve a
            # dictionary with all the properties for this spaxel.
            routeparams = {'name': self.plateifu,
                           'x': self.x, 'y': self.y,
                           'bintype': self.bintype,
                           'template_kin': self.template_kin}

            url = marvin.config.urlmap['api']['getProperties']['url'].format(**routeparams)

            # Make the API call
            response = api.Interaction(url, params={'mplver': self._mplver,
                                                    'drver': self._drver})

            # Temporarily stores the arrays prior to subclassing from np.array
            data = response.getData()

            properties = {}
            for prop_fullname in data['properties']:
                prop = data['properties'][prop_fullname]
                properties[prop_fullname] = AnalysisProperty(
                    prop['name'],
                    channel=prop['channel'],
                    value=prop['value'],
                    ivar=prop['ivar'],
                    mask=prop['mask'],
                    unit=prop['unit'],
                    description=prop['description'])

        self.properties = DictOfProperties(properties)

    def _load_models(self):

        assert self.modelcube, 'a ModelCube is needed to initialise models.'

        if self.modelcube.data_origin == 'file':

            hdus = self.modelcube.data
            flux_array = hdus['FLUX'].data[:, self.y, self.x]
            flux_ivar = hdus['IVAR'].data[:, self.y, self.x]
            flux_mask = hdus['MASK'].data[:, self.y, self.x]
            model_array = hdus['MODEL'].data[:, self.y, self.x]
            model_emline = hdus['EMLINE'].data[:, self.y, self.x]
            model_emline_base = hdus['EMLINE_BASE'].data[:, self.y, self.x]
            model_emline_mask = hdus['EMLINE_MASK'].data[:, self.y, self.x]

        elif self.modelcube.data_origin == 'db':

            if marvin.marvindb is None:
                raise MarvinError('there is not a valid DB connection.')

            session = marvin.marvindb.session
            dapdb = marvin.marvindb.dapdb

            modelcube_db_spaxel = session.query(dapdb.ModelSpaxel).filter(
                dapdb.ModelSpaxel.modelcube == self.modelcube.data,
                dapdb.ModelSpaxel.x == self.x, dapdb.ModelSpaxel.y == self.y).one()

            if modelcube_db_spaxel is None:
                raise MarvinError('cannot find an modelcube spaxel for '
                                  'x={0.x}, y={0.y}'.format(self))

            flux_array = modelcube_db_spaxel.flux
            flux_ivar = modelcube_db_spaxel.ivar
            flux_mask = modelcube_db_spaxel.mask
            model_array = modelcube_db_spaxel.model
            model_emline = modelcube_db_spaxel.emline
            model_emline_base = modelcube_db_spaxel.emline_base
            model_emline_mask = modelcube_db_spaxel.emline_mask

        elif self.modelcube.data_origin == 'api':

            # Calls /modelcubes/<name>/models/<path:path> to retrieve a
            # dictionary with all the models for this spaxel.
            url = marvin.config.urlmap['api']['getModels']['url']
            url_full = url.format(name=self.plateifu,
                                  bintype=self.bintype,
                                  template_kin=self.template_kin,
                                  x=self.x, y=self.y)

            try:
                response = api.Interaction(url_full, params={'mplver': self._mplver,
                                                             'drver': self._drver})
            except Exception as ee:
                raise MarvinError('found a problem when checking if remote model cube '
                                  'exists: {0}'.format(str(ee)))

            data = response.getData()

            flux_array = np.array(data['flux_array'])
            flux_ivar = np.array(data['flux_ivar'])
            flux_mask = np.array(data['flux_mask'])
            model_array = np.array(data['model_array'])
            model_emline = np.array(data['model_emline'])
            model_emline_base = np.array(data['model_emline_base'])
            model_emline_mask = np.array(data['model_emline_mask'])

        # Instantiates the model attributes.

        self.redcorr = Spectrum(self.modelcube.redcorr,
                                wavelength=self.modelcube.wavelength,
                                wavelength_unit='Angstrom')

        self.model_flux = Spectrum(flux_array,
                                   units='1E-17 erg/s/cm^2/Ang/spaxel',
                                   wavelength=self.modelcube.wavelength,
                                   wavelength_unit='Angstrom',
                                   ivar=flux_ivar,
                                   mask=flux_mask)

        self.model = Spectrum(model_array,
                              units='1E-17 erg/s/cm^2/Ang/spaxel',
                              wavelength=self.modelcube.wavelength,
                              wavelength_unit='Angstrom')

        self.emline = Spectrum(model_emline,
                               units='1E-17 erg/s/cm^2/Ang/spaxel',
                               wavelength=self.modelcube.wavelength,
                               wavelength_unit='Angstrom',
                               mask=model_emline_mask)

        self.emline_base = Spectrum(model_emline_base,
                                    units='1E-17 erg/s/cm^2/Ang/spaxel',
                                    wavelength=self.modelcube.wavelength,
                                    wavelength_unit='Angstrom',
                                    mask=model_emline_mask)

        self.stellar_continuum = Spectrum(
            self.model.flux - self.emline.flux - self.emline_base.flux,
            units='1E-17 erg/s/cm^2/Ang/spaxel',
            wavelength=self.modelcube.wavelength,
            wavelength_unit='Angstrom',
            mask=model_emline_mask)
