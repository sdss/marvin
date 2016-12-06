#!/usr/bin/env python
# encoding: utf-8
#
# rss.py
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

from marvin.core.core import MarvinToolsClass
from marvin.core.exceptions import MarvinError, MarvinUserWarning

from marvin.tools.spectrum import Spectrum


class RSS(MarvinToolsClass, list):
    """A class to interface with MaNGA RSS data.

    This class represents a fully reduced RSS file, initialised either
    from a file, a database, or remotely via the Marvin API. The class
    inherits from Python's list class, and is defined as a list of
    RSSFiber objects.

    Parameters:
        filename (str):
            The path of the file containing the RSS to load.
        mangaid (str):
            The mangaid of the RSS to load.
        plateifu (str):
            The plate-ifu of the RSS to load (either ``mangaid`` or
            ``plateifu`` can be used, but not both).
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See
            :doc:`Mode secision tree</mode_decision>`.
        nsa_source ({'auto', 'drpall', 'nsa'}):
            Defines how the NSA data for this object should loaded when
            ``RSS.nsa`` is first called. If ``drpall``, the drpall file will
            be used (note that this will only contain a subset of all the NSA
            information); if ``nsa``, the full set of data from the DB will be
            retrieved. If the drpall file or a database are not available, a
            remote API call will be attempted. If ``nsa_source='auto'``, the
            source will depend on how the ``RSS`` object has been
            instantiated. If the cube has ``RSS.data_origin='file'``,
            the drpall file will be used (as it is more likely that the user
            has that file in their system). Otherwise, ``nsa_source='nsa'``
            will be assumed. This behaviour can be modified during runtime by
            modifying the ``RSS.nsa_mode`` with one of the valid values.
        release (str):
            The MPL/DR version of the data to use.

    Return:
        rss:
            An object representing the RSS entity. The object is a list of
            RSSFiber objects, one for each fibre in the RSS entity.

    """

    def __init__(self, *args, **kwargs):

        valid_kwargs = ['filename', 'mangaid', 'plateifu', 'mode',
                        'drpall', 'release', 'nsa_source']

        assert len(args) == 0, 'RSS does not accept arguments, only keywords.'
        for kw in kwargs:
            assert kw in valid_kwargs, 'keyword {0} is not valid'.format(kw)

        self.data = None

        MarvinToolsClass.__init__(self, *args, **kwargs)

        if self.data_origin == 'file':
            self._load_rss_from_file()
        elif self.data_origin == 'db':
            self._load_rss_from_db()
        elif self.data_origin == 'api':
            self._load_rss_from_api()

        _fibers = self._init_fibers()
        list.__init__(self, _fibers)

        # TODO: check that the drpver of the loaded data matches the one in the object.

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

    def _load_rss_from_file(self):
        """Initialises the RSS object from a file."""

        try:
            self.data = fits.open(self.filename)
            self.mangaid = self.data[0].header['MANGAID'].strip()
            self.plateifu = '{0}-{1}'.format(
                self.data[0].header['PLATEID'], self.data[0].header['IFUDSGN'])
        except Exception as ee:
            raise MarvinError('Could not initialize via filename: {0}'.format(ee))

        # Checks and populates release.
        file_drpver = self.data[0].header['VERSDRP3']
        file_drpver = 'v1_5_1' if file_drpver == 'v1_5_0' else file_drpver

        file_ver = marvin.config.lookUpRelease(file_drpver)
        assert file_ver is not None, 'cannot find file version.'

        if file_ver != self._release:
            warnings.warn('mismatch between file version={0} and object release={1}. '
                          'Setting object release to {0}'.format(file_ver, self._release),
                          MarvinUserWarning)
            self._release = file_ver

        self._drpver, self._dapver = marvin.config.lookUpVersions(release=self._release)

    def _load_rss_from_db(self):
        """Initialises the RSS object from the DB."""

        import sqlalchemy

        mdb = marvin.marvindb

        if not mdb.isdbconnected:
            raise MarvinError('No db connected')

        plate, ifudesign = [item.strip() for item in self.plateifu.split('-')]

        try:
            self.data = mdb.session.query(mdb.datadb.RssFiber).join(
                mdb.datadb.Cube, mdb.datadb.PipelineInfo,
                mdb.datadb.PipelineVersion, mdb.datadb.IFUDesign).filter(
                    mdb.datadb.PipelineVersion.version == self._drpver,
                    mdb.datadb.Cube.plate == plate,
                    mdb.datadb.IFUDesign.name == ifudesign).all()

        except sqlalchemy.orm.exc.NoResultFound as ee:
            raise MarvinError('Could not retrieve RSS for plate-ifu {0}: '
                              'No Results Found: {1}'
                              .format(self.plateifu, ee))

        except Exception as ee:
            raise MarvinError('Could not retrieve RSS for plate-ifu {0}: '
                              'Unknown exception: {1}'
                              .format(self.plateifu, ee))

        if not self.data:
            raise MarvinError('Could not retrieve RSS for plate-ifu {0}: '
                              'Unknown error.'.format(self.plateifu))

    def _load_rss_from_api(self):
        """Initialises the RSS object using the remote API."""

        # Checks that the RSS exists.
        routeparams = {'name': self.plateifu}
        url = marvin.config.urlmap['api']['getRSS']['url'].format(**routeparams)

        # Make the API call
        self.ToolInteraction(url)

    def _init_fibers(self):
        """Initialises the object as a list of RSSFiber instances."""

        if self.data_origin == 'file':
            _fibers = [RSSFiber._init_from_hdu(hdulist=self.data, index=ii)
                       for ii in range(self.data[1].data.shape[0])]

        elif self.data_origin == 'db':
            _fibers = [RSSFiber._init_from_db(rssfiber=rssfiber) for rssfiber in self.data]

        elif self.data_origin == 'api':
            # Makes a call to the API to retrieve all the arrays for all the fibres.

            routeparams = {'name': self.plateifu}
            url = marvin.config.urlmap['api']['getRSSAllFibers']['url'].format(**routeparams)

            # Make the API call
            response = self.ToolInteraction(url)
            data = response.getData()

            wavelength = np.array(data['wavelength'])

            _fibers = []
            for ii in range(len(data) - 1):
                flux = np.array(data[str(ii)][0])
                ivar = np.array(data[str(ii)][1])
                mask = np.array(data[str(ii)][2])
                _fibers.append(
                    RSSFiber(flux, ivar=ivar, mask=mask, wavelength=wavelength,
                             mangaid=self.mangaid, plateifu=self.plateifu, data_origin='api'))

        return _fibers


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

    .. |spectrum| replace:: :class:`~marvin.tools.spectrum.Spectrum`

    """

    def __init__(self, *args, **kwargs):

        self.mangaid = kwargs.pop('mangaid', None)
        self.plateifu = kwargs.pop('plateifu', None)
        self.data_origin = kwargs.pop('data_origin', None)

        flux_units = '1e-17 erg/s/cm^2/Ang/fiber'
        wavelength_unit = 'Angstrom'
        kwargs['units'] = flux_units
        kwargs['wavelength_unit'] = wavelength_unit

        Spectrum.__init__(self, *args, **kwargs)

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
