#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2016-04-11
# @Filename: rss.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-30 19:42:17


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
from .mixins import NSAMixIn
from .quantities.spectrum import Spectrum


class RSS(MarvinToolsClass, NSAMixIn, list):
    """A class to interface with a MaNGA DRP row-stacked spectra file.

    This class represents a fully reduced DRP row-stacked spectra object,
    initialised either from a file, a database, or remotely via the Marvin API.
    Instances of `.RSS` are a list of `.RSSFiber` objects, one for each fibre
    and exposure. `.RSSFiber` are initialised lazily, containing only basic
    information. They need to be initialised by calling `.RSSFiber.load`
    (unless `.RSS.autoload` is ``True``, in which case the instance is loaded
    when first accessed).

    In addition to the input arguments supported by `~.MarvinToolsClass` and
    `~.NSAMixIn`, this class accepts an ``autoload`` keyword argument that
    defines whether `.RSSFiber` objects should be automatically loaded when
    they are accessed.

    """

    _qualflag = 'DRP3QUAL'

    def __init__(self, input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None, autoload=True,
                 drpall=None, download=None, nsa_source='auto'):

        MarvinToolsClass.__init__(self, input=input, filename=filename, mangaid=mangaid,
                                  plateifu=plateifu, mode=mode, data=data, release=release,
                                  drpall=drpall, download=download)

        NSAMixIn.__init__(self, nsa_source=nsa_source)

        #: An `astropy.table.Table` with the observing information associated
        #: with this RSS object.
        self.obsinfo = None

        #: If True, unloaded `.RSSFiber` instances are automatically loaded
        #: when accessed. Otherwise, they need to be loaded via `.RSSFiber.load`.
        self.autoload = autoload

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

        # EXPNUM in obsinfo is a string. Cast it to int
        self.obsinfo['EXPNUM'] = self.obsinfo['EXPNUM'].astype(numpy.int32)

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

    def __getitem__(self, fiberid):
        """Returns the `.RSSFiber` whose fiberid matches the input."""

        rssfiber = super(RSS, self).__getitem__(fiberid)

        if self.autoload and not rssfiber.loaded:
            rssfiber.load()

        return rssfiber

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

    def getCube(self):
        """Returns the `~marvin.tools.cube.Cube` associated with this RSS."""

        return Cube(plateifu=self.plateifu, mode=self.mode, release=self.release)

    def load_all(self):
        """Loads all the `.RSSFiber` associated to this `.RSS` instance."""

        for rssfiber in self:
            if not rssfiber.loaded:
                rssfiber.load()

    def select_fibers(self, exposure_no=None, set=None, mjd=None):
        """Selects fibres that match one or multiple of the input parameters.

        Parameters
        ----------
        exposure_no : int
            The exposure number. Ignored if ``None``.
        set : int
            The set id of the exposure. Ignored if ``None``.
        mjd : int
            The MJD of the exposure. Ignored if ``None``.

        Returns
        -------
        rssfibers : list
            A list of `.RSSFiber` instances whose obsinfo matches all the input
            parameters. The `.RSS.autoload` option is respected.

        Example
        -------

            >>> rss = marvin.tools.RSS('8485-1901')
            >>> fibers = rss.select_fibers(set=2)
            >>> fibers
            [<RSSFiber [ 2.22306705, 11.84955406,  9.65761662, ...,  0.        ,
                         0.        ,  0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
            <RSSFiber [2.18669987, 1.4861778 , 2.55065155, ..., 0.        , 0.        ,
                       0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
            <RSSFiber [2.75228763, 5.53485441, 2.31695175, ..., 0.        , 0.        ,
                       0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>]

        """

        mask_exp = (self.obsinfo['EXPNUM'].astype(int) == exposure_no) if exposure_no else True
        mask_set = (self.obsinfo['SET'].astype(int) == set) if set else True
        mask_mjd = (self.obsinfo['MJD'].astype(int) == mjd) if mjd else True

        mask = mask_exp & mask_set & mask_mjd
        valid_exposures = numpy.where(mask)[0]

        n_exposures = len(self.obsinfo)
        n_fibres_per_exposure = self._nfibers // n_exposures
        fibre_to_exposure = numpy.arange(self._nfibers) // n_fibres_per_exposure

        fibres_in_valid_exposures = numpy.where(numpy.in1d(fibre_to_exposure, valid_exposures))[0]

        return [self[ii] for ii in fibres_in_valid_exposures]

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

        n_exposures = len(self.obsinfo)
        n_fibres_per_exposure = self._nfibers // n_exposures

        for fiberid in range(self._nfibers):
            exp_index = fiberid // n_fibres_per_exposure
            exp_obsinfo = self.obsinfo[[exp_index]]
            self.append(RSSFiber(fiberid, self, self._wavelength, load=False,
                                 obsinfo=exp_obsinfo, pixmask_flag=self.header['MASKNAME']))


class RSSFiber(Spectrum):
    """A `~astropy.units.Quantity` representing a fibre observation.

    Represents the spectral flux observed though a fibre, and associated with
    an `.RSS` object. In addition to the flux, it contains information about
    the inverse variance, mask, and other associated spectra defined in the
    datamodel.

    Parameters
    ----------
    fiberid : int
        The fiberid (0-indexed row in the parent `.RSS` object) for this fibre
        observation.
    rss : `.RSS`
        The parent `.RSS` object with which this fibre observation is
        associated.
    wavelength : numpy.ndarray
        The wavelength positions of each array element, in Angstrom.
    load : bool
        Whether the information in the `.RSSFiber` should be loaded during
        instantiation. Defaults to lazy loading (use `.RSSFiber.load` to
        load the fibre information).
    obsinfo : astropy.table.Table
        A `~astropy.table.Table` with the information for the exposure to
        which this fibre observation belongs.
    kwargs : dict
        Additional keyword arguments to be passed to `.Spectrum`.

    """

    def __new__(cls, fiberid, rss, wavelength, pixmask_flag=None, load=False,
                obsinfo=None, **kwargs):

        # For now we instantiate a mostly empty Spectrum. Proper instantiation
        # will happen in load().

        array_size = len(wavelength)

        obj = super(RSSFiber, cls).__new__(
            cls, numpy.zeros(array_size, dtype=numpy.float64), wavelength,
            scale=None, unit=None,)

        obj._extra_attributes = ['fiberid', 'rss', 'loaded', 'obsinfo']
        obj._spectra = []

        return obj

    def __init__(self, fiberid, rss, wavelength, pixmask_flag=None, load=False,
                 obsinfo=None, **kwargs):

        self.fiberid = fiberid
        self.rss = rss
        self.obsinfo = obsinfo

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

        # Adds _extra_attributes from the previous object.
        if hasattr(obj, '_extra_attributes'):
            for attr in obj._extra_attributes:
                setattr(self, attr, getattr(obj, attr, None))
        self._extra_attributes = getattr(obj, '_extra_attributes', None)

        # Adds the additional spectra from the previous object.
        if hasattr(obj, '_spectra'):
            for spectrum in obj._spectra:
                setattr(self, spectrum, getattr(obj, spectrum, None))
        self._spectra = getattr(obj, '_spectra', None)

    def __getitem__(self, sl):

        new_obj = super(RSSFiber, self).__getitem__(sl)

        for spectra_name in self._spectra:
            current_spectrum = getattr(self, spectra_name, None)
            new_spectrum = None if current_spectrum is None else current_spectrum.__getitem__(sl)
            setattr(new_obj, spectra_name, new_spectrum)

        return new_obj

    def load(self):
        """Loads the fibre information."""

        assert self.loaded is False, 'object already loaded.'

        # Depending on whether the parent RSS is a file or API-populated, we
        # select the data to use.
        if self.rss.data_origin == 'file':

            # If the data origin is a file we use the HDUList in rss.data
            rss_data = self.rss.data

        elif self.rss.data_origin == 'api':

            # If data origin is the API, we make a request for the data
            # associated with this fiberid for all the extensions in the file.

            url = marvin.config.urlmap['api']['getRSSFiber']['url']

            try:
                response = self.rss._toolInteraction(url.format(name=self.rss.plateifu,
                                                                fiberid=self.fiberid))
            except Exception as ee:
                raise MarvinError('found a problem retrieving RSS fibre data for '
                                  'plateifu={!r}, fiberid={!r}: {}'.format(
                                      self.rss.plateifu, self.fiberid, str(ee)))

            api_data = response.getData()

            # Create a quick and dirty HDUList from the API data so that we
            # can parse it in the same way as if the data origin is file.
            rss_data = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU()])
            for ext in api_data:
                rss_data.append(astropy.io.fits.ImageHDU(data=api_data[ext], name=ext.upper()))

        else:
            raise ValueError('invalid data_origin={!r}'.format(self.rss.data_origin))

        # Compile a list of all RSS datamodel extensions, either RSS or spectra
        datamodel_extensions = self.rss.datamodel.rss + self.rss.datamodel.spectra

        for extension in datamodel_extensions:

            # Retrieve the value (and mask and ivar, if associated) for each extension.
            value, ivar, mask = self._get_extension_data(extension, rss_data,
                                                         data_origin=self.rss.data_origin)

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

    def _get_extension_data(self, extension, data, data_origin='file'):
        """Returns the value of an extension for this fibre, either from file or API.

        Parameters
        ----------
        extension : datamodel object
            The datamodel object containing the information for the extension
            we want to retrieve.
        data : ~astropy.io.fits.HDUList
            An `~astropy.io.fits.HDUList` object containing the RSS
            information.

        """

        # Determine if this is an RSS datamodel object or an spectrum.
        # If the origin is the API, the extension data contains a single spectrum,
        # not a row-stacked array, so we consider it a 1D array.
        is_extension_data_1D = isinstance(extension, SpectrumDataModel) or data_origin == 'api'

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
        if not is_extension_data_1D:
            value = value[self.fiberid, :]
            mask = mask[self.fiberid, :] if mask is not None else None
            ivar = ivar[self.fiberid, :] if ivar is not None else None

        return value, ivar, mask

    @property
    def masked(self):
        """Return a masked array where the mask is greater than zero."""

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
