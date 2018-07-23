#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2016-04-11
# @Filename: rss.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-23 01:05:25


from __future__ import division, print_function

import os
import warnings

import astropy.io.ascii
import astropy.table
import astropy.wcs
from astropy.io import fits

import marvin
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.utils.datamodel.drp import datamodel_rss

from .core import MarvinToolsClass
from .cube import Cube
from .mixins import GetApertureMixIn, NSAMixIn
from .quantities.spectrum import Spectrum


class RSS(MarvinToolsClass, NSAMixIn, GetApertureMixIn):
    """A class to interface with a MaNGA DRP row-stacked spectra file.

    This class represents a fully reduced DRP data cube, initialised either
    from a file, a database, or remotely via the Marvin API.

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


class RSSFiber(Spectrum):
    """A class to represent a MaNGA RSS fiber.

    This class is basically a subclass of |spectrum| with additional
    functionality. It is not intended to be initialised directly, but via
    the :py:meth:`RSS._initFibers` method.

    Parameters:
        args:
            Arguments to pass to |spectrum| for initialisation.
        kwargs:
            Keyword arguments to pass to |spectrum| for initialisation.

    Return:
        rssfiber:
            An object representing the RSS fiber entity.

    .. |spectrum| replace:: :class:`~marvin.tools.quantities.Spectrum`

    """

    def __init__(self, *args, **kwargs):

        self.mangaid = kwargs.pop('mangaid', None)
        self.plateifu = kwargs.pop('plateifu', None)
        self.data_origin = kwargs.pop('data_origin', None)

        flux_units = '1e-17 erg/s/cm^2/Ang/fiber'
        wavelength_unit = 'Angstrom'
        kwargs['units'] = flux_units
        kwargs['wavelength_unit'] = wavelength_unit

        # Spectrum.__init__(self, **kwargs)

    def __repr__(self):
        """Representation for RSSFiber."""

        return ('<Marvin RSSFiber (mangaid={self.mangaid!r}, plateifu={self.plateifu!r}, '
                'data_origin={self.data_origin!r})>'.format(self=self))

    @classmethod
    def _init_from_hdu(cls, hdulist, index):
        """Initialises a RSSFiber object from a RSS HDUList."""

        assert index is not None, \
            'if hdu is defined, an index is required.'

        mangaid = hdulist[0].header['MANGAID'].strip()
        plateifu = '{0}-{1}'.format(hdulist[0].header['PLATEID'], hdulist[0].header['IFUDSGN'])

        flux = hdulist['FLUX'].data[index, :]
        ivar = hdulist['IVAR'].data[index, :]
        mask = hdulist['MASK'].data[index, :]
        wave = hdulist['WAVE'].data

        obj = RSSFiber(flux, ivar=ivar, mask=mask, wavelength=wave, mangaid=mangaid,
                       plateifu=plateifu, data_origin='file')

        return obj

    @classmethod
    def _init_from_db(cls, rssfiber):
        """Initialites a RSS fiber from the DB."""

        mangaid = rssfiber.cube.mangaid
        plateifu = '{0}-{1}'.format(rssfiber.cube.plate, rssfiber.cube.ifu.name)

        obj = RSSFiber(rssfiber.flux, ivar=rssfiber.ivar, mask=rssfiber.mask,
                       wavelength=rssfiber.cube.wavelength.wavelength, mangaid=mangaid,
                       plateifu=plateifu, data_origin='db')

        return obj
