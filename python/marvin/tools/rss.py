#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2016-04-11
# @Filename: rss.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-23 21:16:07


from __future__ import division, print_function

import os
import warnings

import astropy.io.ascii
import astropy.table
import astropy.units
import astropy.wcs
import numpy
from astropy.io import fits

import marvin
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.utils.datamodel.drp import datamodel_rss
from marvin.utils.datamodel.drp.base import Spectrum as SpectrumDataModel

from .core import MarvinToolsClass
from .cube import Cube
from .mixins import GetApertureMixIn, NSAMixIn
from .quantities.spectrum import Spectrum


class RSS(MarvinToolsClass, NSAMixIn, GetApertureMixIn, list):
    """A class to interface with a MaNGA DRP row-stacked spectra file.

    This class represents a fully reduced DRP row-stacked spectra object,
    initialised either from a file, a database, or remotely via the Marvin API.

    See `~.MarvinToolsClass` and `~.NSAMixIn` for a list of input parameters.

    """

    def __init__(self, input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None,
                 drpall=None, download=None, nsa_source='auto'):

        MarvinToolsClass.__init__(self, input=input, filename=filename, mangaid=mangaid,
                                  plateifu=plateifu, mode=mode, data=data, release=release,
                                  drpall=drpall, download=download)

        NSAMixIn.__init__(self, nsa_source=nsa_source)

        #: An `astropy.table.Table` with the observing information associated
        #: with this RSS object.
        self.obsinfo = None

        if self.data_origin == 'file':
            self._load_rss_from_file(data=self.data)
        elif self.data_origin == 'db':
            self._load_rss_from_db(data=self.data)
        elif self.data_origin == 'api':
            self._load_rss_from_api()

        Cube._init_attributes(self)

        # Checks that the drpver set in MarvinToolsClass matches the header
        header_drpver = self.header['VERSDRP3'].strip()
        header_drpver = 'v1_5_1' if header_drpver == 'v1_5_0' else header_drpver
        assert header_drpver == self._drpver, ('mismatch between cube._drpver={0} '
                                               'and header drpver={1}'.format(self._drpver,
                                                                              header_drpver))

        # Inits self as an empty list.
        list.__init__(self, [])
        self._populate_fibres()

    def _set_datamodel(self):
        """Sets the datamodel for DRP."""

        self.datamodel = datamodel_rss[self.release.upper()]
        self._bitmasks = datamodel_rss[self.release.upper()].bitmasks

    def __repr__(self):
        """Representation for RSS."""

        return ('<Marvin RSS (mangaid={self.mangaid!r}, plateifu={self.plateifu!r}, '
                'mode={self.mode!r}, data_origin={self.data_origin!r})>'.format(self=self))

    def _getFullPath(self):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return super(RSS, self)._getFullPath('mangarss', ifu=ifu, drpver=self._drpver, plate=plate)

    def download(self):
        """Downloads the cube using sdss_access - Rsync"""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return super(RSS, self).download('mangarss', ifu=ifu, drpver=self._drpver, plate=plate)

    def _load_rss_from_file(self, data=None):
        """Initialises the RSS object from a file."""

        if data is not None:
            assert isinstance(data, fits.HDUList), 'data is not an HDUList object'
        else:
            try:
                self.data = fits.open(self.filename)
            except (IOError, OSError) as err:
                raise OSError('filename {0} cannot be found: {1}'.format(self.filename, err))

        self.header = self.data[1].header
        self.wcs = astropy.wcs.WCS(self.header)
        self.wcs = self.wcs.dropaxis(1)  # The header creates an empty axis for the exposures.

        # Confirm that this is a RSS file
        assert 'XPOS' in self.data and self.header['CTYPE1'] == 'WAVE-LOG', \
            'invalid file type. It does not appear to be a LOGRSS.'

        self._wavelength = self.data['WAVE'].data
        self._shape = None
        self._nfibers = self.data['FLUX'].shape[0]

        self.obsinfo = astropy.table.Table(self.data['OBSINFO'].data)

        Cube._do_file_checks(self)

    def _load_rss_from_db(self, data=None):
        """Initialises the RSS object from the DB.

        At this time the DB does not contain enough information to successfully
        instantiate a RSS object so we hack the data access mode to try to use
        files. For users this should be irrelevant since they rarely will have
        a Marvin DB. For the API, it means the access to RSS data will happen
        via files.

        """

        warnings.warn('DB mode is not working for RSS. Trying file access mode.',
                      MarvinUserWarning)

        fullpath = self._getFullPath()
        if fullpath and os.path.exists(fullpath):
            self.filename = fullpath
            self.data_origin = 'file'
            self._load_rss_from_file()
        else:
            raise MarvinError('cannot find a valid RSS file for '
                              'plateifu={self.plateifu!r}, release={self.release!r}'
                              .format(self=self))

    def _load_rss_from_api(self):
        """Initialises the RSS object using the remote API."""

        # Checks that the RSS exists.
        routeparams = {'name': self.plateifu}
        url = marvin.config.urlmap['api']['getRSS']['url'].format(**routeparams)

        try:
            response = self._toolInteraction(url.format(name=self.plateifu))
        except Exception as ee:
            raise MarvinError('found a problem when checking if remote RSS '
                              'exists: {0}'.format(str(ee)))

        data = response.getData()

        self.header = fits.Header.fromstring(data['header'])
        self.wcs = astropy.wcs.WCS(fits.Header.fromstring(data['wcs_header']))
        self._wavelength = data['wavelength']
        self._nfibers = data['nfibers']

        self.obsinfo = astropy.io.ascii.read(data['obsinfo'])

        if self.plateifu != data['plateifu']:
            raise MarvinError('remote RSS has a different plateifu!')

        return

    def _populate_fibres(self):
        """Populates the internal list of fibres."""

        for ii in range(self._nfibers):
            self.append(RSSFiber(ii, self, self._wavelength, load=False,
                                 pixmask_flag=self.header['MASKNAME']))


class RSSFiber(Spectrum):

    def __new__(cls, fiberid, rss, wavelength, **kwargs):

        # For now we instantiate a mostly empty Spectrum. Proper instantiation
        # will happen in load().

        array_size = len(wavelength)

        obj = super(RSSFiber, cls).__new__(
            cls, numpy.zeros(array_size, dtype=numpy.float64), wavelength,
            scale=None, unit=None)

        obj._extra_attributes = ['fiberid', 'rss', 'loaded']
        obj._spectra = []

        return obj

    def __init__(self, fiberid, rss, wavelength, pixmask_flag=None, load=False):

        self.fiberid = fiberid
        self.rss = rss

        self.pixmask_flag = pixmask_flag

        self.loaded = False
        if load:
            self.load()

    def __repr__(self):

        if not self.loaded:
            return ('<RSSFiber (plateifu={self.rss.plateifu!r}, '
                    'fiberid={self.fiberid!r}, loaded={self.loaded!r})>'.format(self=self))
        else:
            return super(RSSFiber, self).__repr__()

    def __array_finalize__(self, obj):

        if obj is None:
            return

        super(RSSFiber, self).__array_finalize__(obj)

        if hasattr(obj, '_extra_attributes'):
            for attr in obj._extra_attributes:
                setattr(self, attr, getattr(obj, attr, None))
        self._extra_attributes = getattr(obj, '_extra_attributes', None)

        if hasattr(obj, '_spectra'):
            for spectrum in obj._spectra:
                setattr(self, spectrum, getattr(obj, spectrum, None))
        self._spectra = getattr(obj, '_spectra', None)

    def __getitem__(self, sl):

        new_obj = super(RSSFiber, self).__getitem__(sl)

        for spectra_name in self._spectra:

            current_value = getattr(self, spectra_name, None)

            if current_value is None:
                new_value = None
            else:
                new_value = current_value.__getitem__(sl)

            setattr(new_obj, spectra_name, new_value)

        return new_obj

    def load(self):
        """Loads the fibre information."""

        assert self.loaded is False, 'object already loaded.'

        datamodel_extensions = self.rss.datamodel.rss + self.rss.datamodel.spectra

        for extension in datamodel_extensions:

            value, ivar, mask = self._get_extension_data(extension)

            if extension.name == 'flux':

                self.value[:] = value[:]
                self.ivar = ivar
                self.mask = mask
                self._set_unit(extension.unit)

            else:

                new_spectrum = Spectrum(value, self.wavelength, ivar=ivar, mask=mask,
                                        unit=extension.unit)
                setattr(self, extension.name, new_spectrum)

                self._spectra.append(extension.name)

        self.loaded = True

    @property
    def masked(self):
        """Return a masked array.

        If the `~QuantityMixIn.pixmask` is set, and the maskbit contains the
        ``DONOTUSE`` and ``NOCOV`` labels, the returned array will be masked
        for the values containing those bits. Otherwise, all values where the
        mask is greater than zero will be masked.

        """

        assert self.mask is not None, 'mask is not set'

        return numpy.ma.array(self.value, mask=(self.mask > 0))

    def descale(self):
        """Returns a copy of the object in which the scale is unity.

        Note that this only affects to the core value of this quantity.
        Associated array attributes will not be modified.

        Example:

            >>> fiber.unit
            Unit("1e-17 erg / (Angstrom cm2 fiber s)")
            >>> fiber[100]
            <RSSFiber 0.270078063011169 1e-17 erg / (Angstrom cm2 fiber s)>
            >>> fiber_descaled = fiber.descale()
            >>> fiber_descaled.unit
            Unit("Angstrom cm2 fiber s")
            >>> fiber[100]
            <RSSFiber 2.70078063011169e-18 erg / (Angstrom cm2 fiber s)>

        """

        if self.unit.scale == 1:
            return self

        value_descaled = self.value * self.unit.scale
        value_unit = astropy.units.CompositeUnit(1, self.unit.bases, self.unit.powers)

        if self.ivar is not None:
            ivar_descaled = self.ivar / (self.unit.scale ** 2)
        else:
            ivar_descaled = None

        copy_of_self = self.copy()
        copy_of_self.value[:] = value_descaled
        copy_of_self.ivar = ivar_descaled
        copy_of_self._set_unit(value_unit)

        return copy_of_self

    def _get_extension_data(self, extension):
        """Returns the value of an extension for this fibre, either from file or API.

        Parameters
        ----------
        extension : datamodel object
            The datamodel object containing the information for the extension
            we want to retrieve.

        """

        if self.rss.data_origin == 'file':

            data = self.rss.data

            # Determine if this is an RSS datamodel object or an spectrum.
            is_spectrum = isinstance(extension, SpectrumDataModel)

            value = data[extension.fits_extension()].data

            if extension.has_mask():
                mask = data[extension.fits_extension('mask')].data
            else:
                mask = None

            if hasattr(extension, 'has_ivar') and extension.has_ivar():
                ivar = data[extension.fits_extension('ivar')].data
            elif hasattr(extension, 'has_std') and extension.has_std():
                std = data[extension.fits_extension('std')].data
                ivar = 1. / (std**2)
            else:
                ivar = None

            # If this is an RSS, gets the right row in the stacked spectra.
            if not is_spectrum:
                value = value[self.fiberid, :]
                mask = mask[self.fiberid, :] if mask is not None else None
                ivar = ivar[self.fiberid, :] if ivar is not None else None

            return value, ivar, mask
