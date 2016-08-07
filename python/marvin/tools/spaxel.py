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

from astropy.io import fits
import numpy as np

import marvin
from marvin.api import api
from marvin.core import MarvinToolsClass
from marvin.core import MarvinError
import marvin.core.exceptions
import marvin.tools.maps
from marvin.tools.spectrum import Spectrum
import marvin.tools.analysis_props
import marvin.core.core
import marvin.utils.general.dap
import marvin.utils.general.general


class Spaxel(MarvinToolsClass):
    """A class to interface with a spaxel in a cube.

    This class represents a fully reduced spaxel, initialised either
    from a file, a database, or remotely via the Marvin API. An spaxel contains
    flux, ivar, and mask information. Additionally, the spaxel can be
    initialised with DAP properties if the ``create_dap_properties=True``
    keyword is passed. This is done by default if the spaxel is loaded
    during the initialisation of a ``Maps`` object.

    Parameters:
        x,y (int):
            The `x` and `y` coordinates of the spaxel in the cube (0-indexed).
        filename (str):
            The path of the data cube file containing the spaxel to load.
        mangaid (str):
            The mangaid of the spaxel to load.
        plateifu (str):
            The plate-ifu of the spaxel to load (either ``mangaid`` or
            ``plateifu`` can be used, but not both).
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See
            :doc:`Mode secision tree</mode_decision>`.
        create_dap_properties (bool):
            If ``True``, populates the object with the DAP ``AnalysisProperty``
            values. Requires defining the ``bintype`` and ``niter``. If one of
            those is not defined, the default map will be used.
        bintype (str or None):
            The binning type of the DAP MAPS file to use. The default value is
            ``'NONE'``
        niter (int or None):
            The iteration number of the DAP map.
        drpall (str):
            The path to the drpall file to use. Defaults to
            ``marvin.config.drpall``.
        drpver (str):
            The DRP version to use. Defaults to ``marvin.config.drpver``.
        dapver (str):
            The DAP version to use. Defaults to ``marvin.confg.dapver``.

    Attributes:
        drp (:class:`~marvin.tools.spectrum.Spectrum` object):
            A `Spectrum` object with the DRP spectrum and associated ivar and
            mask for this spaxel.
        dap (dict):
            A dictionary of `AnalysisProperty` objects, sorted by category
            and name (usually the channel for each category).

    """

    def __new__(cls, *args, **kwargs):

        obj = super(Spaxel, cls).__new__(cls)

        obj.data_origin = None
        obj._hduList = None
        obj._spaxel_db = None
        obj.filename = None
        obj.plateifu = None

        obj.x = None
        obj.y = None

        obj.spectrum = None
        obj.properties = {}

        return obj

    def __init__(self, *args, **kwargs):

        if len(args) > 0:
            self.x = args[0]
            self.y = args[1]
        else:
            self.x = kwargs.pop('x', None)
            self.y = kwargs.pop('y', None)

        assert self.x is not None and self.y is not None

        MarvinToolsClass.__init__(self, *args, **kwargs)

        if self.data_origin == 'file':
            self._getSpaxelFromFile()
        elif self.data_origin == 'db':
            self._getSpaxelFromDB()
        elif self.data_origin == 'api':
            self._getSpaxelFromAPI()

        self._createSpectrum()

        create_dap_properties = kwargs.pop('create_dap_properties', False)

        if create_dap_properties is True:
            self._create_dap_properties(**kwargs)

    def __repr__(self):
        """Spaxel representation."""

        return '<Marvin Spaxel (x={0:d}, y={1:d})>'.format(self.x, self.y)

    def _initDAP(self, data):
        """Initialises the dictionary of `AnalysisProperty` objects.

        Parameters:
            data (dict):
                A dictionary in the form
                `{category: {channel_1: {value: VALUE, ivar: IVAR, mask: MASK},
                             channel_2: ..., unit: UNIT},
                  category_2: {...}, ...}`
                where `category` is of the form `'EMLINE_GFLUX',
                'STELLAR_VEL', ...`, `channels` are the channels defined for
                each category, and `unit` are the physical units for the
                specified values in a category.

        """

        properties = {}
        for cat in data:
            for channel in data[cat]:
                # Skips unit, which is not a real channel.
                if channel == 'unit':
                    continue
                value = data[cat][channel]['value']
                ivar = data[cat][channel]['ivar']
                mask = data[cat][channel]['mask']
                unit = data[cat]['unit']

                analysis_prop_key = str(cat.lower())
                if channel.lower() != 'na':
                    analysis_prop_key += str('_' + channel.lower())

                properties[analysis_prop_key] = marvin.tools.analysis_props.AnalysisProperty(
                    cat.lower(), channel.lower(), value, ivar=ivar, mask=mask, unit=unit)

        self.properties = marvin.core.core.DotableCaseInsensitive(properties)

    def _create_dap_properties(self, **kwargs):
        """Creates the DAP `AnalysisProperty` dictionary.

        This method populates the object dictionary with DAP properties. It
        creates a data dictionary to be passed to `Spaxel._initDAP()` by using
        the same data data access mode used to initialise the spaxel. If
        `dapver` was specified during the spaxel initialisation, uses that;
        otherwise uses the system-wide `dapver`.

        """

        # If the data origin is file or DB, loads the Maps object with the
        # same input arguments and uses Maps.getSpaxel.

        # TODO: This is the easy way to do it, but it may be not the most
        # efficient. We may want to do our own query here to speed things up. (JSG)
        if self.data_origin == 'file' or self.data_origin == 'db':
            maps = marvin.tools.maps.Maps(**kwargs)
            spaxel = maps.getSpaxel(x=self.x, y=self.y, xyorig='lower')
            self.properties = spaxel.properties

        # If the data origin is API, gets the dictionary of DAP properties.
        elif self.data_origin == 'api':
            dict_of_dap_props = marvin.utils.general.dap.get_dict_of_props_api(
                self.plateifu, self.x, self.y)
            self._initDAP(dict_of_dap_props)

    @property
    def categories(self):
        """Returns a list of DAP property categories."""
        return sorted(self.properties.keys())

    def _getFullPath(self, data_type='mangacube', **kwargs):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return MarvinToolsClass._getFullPath(self, data_type, ifu=ifu,
                                             drpver=self._drpver,
                                             plate=plate)

    def download(self, **kwargs):
        """Downloads the cube using sdss_access - Rsync"""
        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return super(Spaxel, self).download(self, 'mangacube', ifu=ifu,
                                            drpver=self._drpver,
                                            plate=plate)

    def _createSpectrum(self):
        """Initialises the :class:`.Spectrum` object based on this Spaxel."""

        if self.data_origin == 'file':
            flux = self._hduList['FLUX'].data
            ivar = self._hduList['IVAR'].data
            mask = self._hduList['MASK'].data
            wavelength = self._hduList['WAVE'].data

        elif self.data_origin == 'db':
            flux = self._spaxel_db.flux
            ivar = self._spaxel_db.ivar
            mask = self._spaxel_db.mask
            wavelength = self._spaxel_db.cube.wavelength.wavelength

        elif self.data_origin == 'api':
            flux = np.array(self._arrays['flux'])
            ivar = np.array(self._arrays['ivar'])
            mask = np.array(self._arrays['mask'])
            wavelength = np.array(self._arrays['wavelength'])

        self.spectrum = Spectrum(flux,
                                 ivar=ivar,
                                 mask=mask,
                                 wavelength=wavelength,
                                 flux_units='1e-17 erg/s/cm^2/Ang/spaxel',
                                 wavelength_unit='Angstrom')

    def _getSpaxelFromFile(self, cubeHDU=None):
        """Initialises the Spaxel object from a file data cube.

        We create a new HDUList object containing all the extensions of the
        cube file except the broand band imaging. For the extension with data
        cubes (flux, ivar, mask), we keep only the spectrum for the given
        spaxel.

        """

        validExts = ['PRIMARY', 'FLUX', 'IVAR', 'MASK', 'WAVE', 'SPECRES',
                     'SPECRESD']

        try:
            if cubeHDU is None:
                cubeHDU = fits.open(self.filename)

            self._hduList = fits.HDUList()

            for ext in cubeHDU:
                if ext.name.upper() in validExts:
                    if ext.data is None or len(ext.data.shape) == 1:
                        self._hduList.append(ext)
                    else:
                        spectrum = ext.data[:, self.y, self.x]
                        newHDU = fits.ImageHDU(data=spectrum, header=ext.header)
                        self._hduList.append(newHDU)

            self.data_origin = 'file'

        except Exception as ee:
            raise MarvinError('Could not initialize via filename: {0}'
                              .format(ee))

    def _getSpaxelFromDB(self, data=None):
        """Initialises the spaxel object from the DB."""

        self.data_origin = 'db'

        if data is None:
            pass
        elif data.__tablename__ == 'spaxel':
            self._spaxel_db = data
            return
        elif data.__tablename__ == 'cube':
            self._spaxel_db = marvin.utils.general.general.getSpaxelXY(
                data, self.plateifu, self.x, self.y)
            return
        else:
            warnings.warn('Spaxel._getSpaxelFromDB: '
                          'data of type {0} is not valid.'.format(type(data)),
                          marvin.core.exceptions.MarvinUserWarning)

        # Performs a complete query.

        mdb = marvin.marvindb

        if not mdb.isdbconnected:
            raise RuntimeError('No db connected')

        plate, ifudesign = map(lambda xx: xx.strip(), self.plateifu.split('-'))

        try:
            self._spaxel_db = mdb.session.query(mdb.datadb.Spaxel).join(
                mdb.datadb.Cube, mdb.datadb.PipelineInfo,
                mdb.datadb.PipelineVersion, mdb.datadb.IFUDesign).filter(
                    mdb.datadb.PipelineVersion.version == self._drpver,
                    mdb.datadb.Cube.plate == plate,
                    mdb.datadb.IFUDesign.name == ifudesign,
                    mdb.datadb.Spaxel.x == int(self.x),
                    mdb.datadb.Spaxel.y == int(self.y)).one()

        except Exception as ee:
            raise MarvinError('Could not retrieve spaxel for plate-ifu {0}. {1}: {2}'
                              .format(self.plateifu, str(ee.__class__.__name__), str(ee)))

    def _getSpaxelFromAPI(self):
        """Initialises the spaxels object using the remote API."""

        # Checks that the spaxel exists.
        routeparams = {'name': self.plateifu,
                       'path': 'x={0}/y={1}'.format(self.x, self.y)}
        url = marvin.config.urlmap['api']['getSpaxel']['url'].format(
            **routeparams)

        # Make the API call
        response = api.Interaction(url)

        # Temporarily stores the arrays prior to subclassing from np.array
        self._arrays = response.getData()

        return response

    @classmethod
    def _initFromData(cls, plateifu, x, y, maps=None, cube=None):
        """Initialises a spaxel from and HDUList or DB object."""

        # TODO: I don't love this implementation. Maybe this can be refactored (JSG)

        assert maps is not None or cube is not None

        obj = Spaxel.__new__(Spaxel)

        obj.x = x
        obj.y = y

        obj.plateifu = plateifu

        # DRP data loading section
        if cube is None:
            pass
        elif cube.data_origin == 'file':
            # Case when we receive a FITS file (open)
            obj._getSpaxelFromFile(cubeHDU=cube.data)
            obj._createSpectrum()
        elif cube.data_origin == 'db':
            # Case when we receive a DB object
            obj._getSpaxelFromDB(data=cube.data)
            obj._createSpectrum()
        elif cube.data_origin == 'api':
            obj.data_origin = 'api'
            obj._getSpaxelFromAPI()
            obj._createSpectrum()
        else:
            raise MarvinError(
                'cannot initialise a Spaxel from data type {0}'.format(type(cube.data)))

        # DAP data loading section
        if maps is None:
            pass
        elif maps.data_origin == 'file':
            dap_dict = marvin.utils.general.dap.maps_file2dict_of_props(maps.data, obj.x, obj.y)
            obj._initDAP(dap_dict)
        elif maps.data_origin == 'db':
            dap_dict = marvin.utils.general.dap.maps_db2dict_of_props(maps.data, obj.x, obj.y)
            obj._initDAP(dap_dict)
        elif maps.data_origin == 'api':
            obj.data_origin = 'api'
            obj._create_dap_properties()
        else:
            raise MarvinError(
                'cannot initialise a Spaxel from data type {0}'.format(type(maps.data)))

        return obj
