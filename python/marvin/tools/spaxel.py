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
import marvin.utils.general.dap
import marvin.utils.general.general
import marvin.tools.cube
import marvin.tools.maps

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
            The path of the data cube file containing the spaxel to load.
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
        logcube (:class:`marvin.tools.maps.LogCube` object or bool)
            As ``cube`` but populates ``Spaxel.model_spectrum`` with an
            spectrum of the corresponding spaxel of the DAP logcube that
            matches ``bintype``, ``template_kin``, and ``template_pop``.
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
        drpall (str):
            The path to the drpall file to use. Defaults to
            ``marvin.config.drpall``.
        drpver (str):
            The DRP version to use. Defaults to ``marvin.config.drpver``.
        dapver (str):
            The DAP version to use. Defaults to ``marvin.config.dapver``.

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
        model_spectrum (:class:`~marvin.tools.spectrum.Spectrum` object):
            A `Spectrum` object with the DAP model spectrum from the logcube.

    """

    def __init__(self, *args, **kwargs):

        valid_kwargs = [
            'x', 'y', 'cube_filename', 'maps_filename', 'mangaid', 'plateifu',
            'cube', 'maps', 'logcube', 'bintype', 'template_kin', 'template_pop',
            'drpall', 'drpver', 'dapver']

        assert len(args) == 0, 'Spaxel does not accept arguments, only keywords.'
        for kw in kwargs:
            assert kw in valid_kwargs, 'keyword {0} is not valid'.format(kw)

        self.cube = kwargs.pop('cube', True) or False
        self.maps = kwargs.pop('maps', True) or False

        if not self.cube and not self.maps:
            raise MarvinError('either cube or maps must be True or a Marvin Cube or Maps object.')

        self._drpver = kwargs.get('drpver', marvin.config.drpver)
        self._dapver = kwargs.get('dapver', marvin.config.dapver)

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

        cube_filename = kwargs.pop('cube_filename', None)
        maps_filename = kwargs.pop('maps_filename', None)
        mangaid_input = kwargs.pop('mangaid', None)
        plateifu_input = kwargs.pop('plateifu', None)

        # Checks that the cube is correct or load ones if cube == True.
        if not isinstance(self.cube, bool):
            assert isinstance(self.cube, marvin.tools.cube.Cube), \
                'cube is not an instance of marvin.tools.cube.Cube or a boolean.'
        elif self.cube is True:
            self.cube = marvin.tools.cube.Cube(filename=cube_filename, mangaid=mangaid_input,
                                               plateifu=plateifu_input, drpver=self._drpver)
        else:
            self.cube = None

        # If we have a cube, loads information from it.
        if self.cube:
            self.plateifu = str(self.cube.plateifu)
            self.mangaid = str(self.cube.mangaid)
            self._drpver = self.cube._drpver
            self._parent_shape = self.cube.shape

            if mangaid_input is not None and mangaid_input != self.mangaid:
                warnings.warn('input mangaid does not match the cube mangaid. '
                              'Will use the cube mangaid.', MarvinUserWarning)

            if plateifu_input is not None and plateifu_input != self.plateifu:
                warnings.warn('input plateifu does not match the cube plateifu. '
                              'Will use the cube plateifu.', MarvinUserWarning)

            # Loads the spectrum
            self._load_spectrum()

        # We do the same with the Maps
        if not isinstance(self.maps, bool):
            assert isinstance(self.maps, marvin.tools.maps.Maps), \
                'maps is not an instance of marvin.tools.maps.Maps or a boolean.'
        elif self.maps is True:

            # Makes sure we call Maps with a filename or with only one of plateifu or mangaid.
            if maps_filename:
                pass
            else:
                if self.plateifu and self.mangaid:
                    plateifu_input = self.plateifu
                    mangaid_input = None

            self.maps = marvin.tools.maps.Maps(filename=maps_filename,
                                               mangaid=mangaid_input,
                                               plateifu=plateifu_input,
                                               drpver=self._drpver,
                                               dapver=self._dapver)
        else:
            self.maps = None

        if self.maps:

            self._dapver = self.maps._dapver

            # Runs some checks to make sure cube and maps match
            if self.cube:
                if (self.cube.plateifu != self.maps.plateifu or
                        self.cube.mangaid != self.maps.mangaid):
                    raise MarvinError('plateifu or mangaid from cube does not '
                                      'match the one in maps.')
                if self.maps._drpver != self.cube._drpver:
                    raise MarvinError('maps has drpver={0} while cube has {1}'
                                      .format(self.maps._drpver, self.cube._drpver))
            else:
                self.plateifu = self.maps.plateifu
                self.mangaid = self.maps.mangaid
                self._drpver = self.maps._drpver
                self._parent_shape = self.maps.shape

            # Loads the properties
            self._load_properties()

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
                                     flux_units='1E-17 erg/s/cm^2/Ang/spaxel',
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
                                     flux_units='1E-17 erg/s/cm^2/Ang/spaxel',
                                     wavelength=cube_db.wavelength.wavelength,
                                     wavelength_unit='Angstrom',
                                     ivar=spaxel.ivar,
                                     mask=spaxel.mask)

            self.specres = np.array(cube_db.specres)
            self.specresd = None

        elif self.cube.data_origin == 'api':

            routeparams = {'name': self.plateifu,
                           'path': 'x={0}/y={1}'.format(self.x, self.y)}

            url = marvin.config.urlmap['api']['getSpectrum']['url'].format(**routeparams)

            # Make the API call
            response = api.Interaction(url, params={'drpver': self._drpver,
                                                    'dapver': self._dapver})

            # Temporarily stores the arrays prior to subclassing from np.array
            data = response.getData()

            self.spectrum = Spectrum(data['flux'],
                                     flux_units='1E-17 erg/s/cm^2/Ang/spaxel',
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

            routeparams = {'name': self.plateifu,
                           'path': 'x={0}/y={1}'.format(self.x, self.y)}

            url = marvin.config.urlmap['api']['getProperties']['url'].format(**routeparams)

            # Make the API call
            response = api.Interaction(url, params={'drpver': self._drpver,
                                                    'dapver': self._dapver})

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
