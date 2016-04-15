#!/usr/bin/env python
# encoding: utf-8
"""

rss.py

Licensed under a 3-clause BSD license.

Revision history:
    11 Apr 2016 J. Sánchez-Gallego
      Initial version

"""

from __future__ import division
from __future__ import print_function
from marvin.tools.core import MarvinToolsClass
from marvin.tools.core import MarvinError
from astropy.io import fits
from marvin.api import api
import marvin
from marvin.tools.spectrum import Spectrum
import numpy as np


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
        skip_check (bool):
            If True, and ``mode='remote'``, skips the API call to check that
            the cube exists.
        drpall (str):
            The path to the drpall file to use. Defaults to
            ``marvin.config.drpall``.
        drpver (str):
            The DRP version to use. Defaults to ``marvin.config.drpver``.

    Return:
        rss:
            An object representing the RSS entity. The object is a list of
            RSSFiber objects, one for each fibre in the RSS entity.

    """

    def __init__(self, *args, **kwargs):

        self._hdu = None
        self._rss_db = None
        self.data_origin = None
        self._fibers = None

        skip_check = kwargs.get('skip_check', False)

        MarvinToolsClass.__init__(self, *args, **kwargs)

        if self.mode == 'local':
            if self.filename:
                self._getRSSFromFile()
            else:
                self._getRSSFromDB()
        else:
            self.data_origin = 'api'
            if not skip_check:
                self._getRSSFromAPI()

        self._initFibers()

    def __repr__(self):
        """Representation for RSS."""

        return ('<Marvin RSS (mangaid={self.mangaid!r}, plateifu={self.plateifu!r}, '
                'mode={self.mode!r}, data_origin={self.data_origin!r})>'.format(self=self))

    def _getFullPath(self, **kwargs):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return super(RSS, self)._getFullPath('mangarss', ifu=ifu,
                                             drpver=self._drpver, plate=plate)

    def _getRSSFromFile(self):
        """Initialises the RSS object from a file."""

        try:
            self._hdu = fits.open(self.filename)
            self.mangaid = self._hdu[0].header['MANGAID'].strip()
            self.plateifu = '{0}-{1}'.format(
                self._hdu[0].header['PLATEID'], self._hdu[0].header['IFUDSGN'])
            self.data_origin = 'file'
        except Exception as ee:
            raise MarvinError('Could not initialize via filename: {0}'
                              .format(ee))

    def _getRSSFromDB(self):
        """Initialises the RSS object from the DB."""

        import sqlalchemy

        mdb = marvin.marvindb

        if not mdb.isdbconnected:
            raise RuntimeError('No db connected')

        plate, ifudesign = map(lambda xx: xx.strip(),
                               self.plateifu.split('-'))

        try:
            self._rss_db = mdb.session.query(mdb.datadb.RssFiber).join(
                mdb.datadb.Cube, mdb.datadb.PipelineInfo,
                mdb.datadb.PipelineVersion, mdb.datadb.IFUDesign).filter(
                    mdb.datadb.PipelineVersion.version == self._drpver,
                    mdb.datadb.Cube.plate == plate,
                    mdb.datadb.IFUDesign.name == ifudesign).all()

        except sqlalchemy.orm.exc.NoResultFound as ee:
            raise RuntimeError('Could not retrieve RSS for plate-ifu {0}: '
                               'No Results Found: {1}'
                               .format(self.plateifu, ee))

        except Exception as ee:
            raise RuntimeError('Could not retrieve RSS for plate-ifu {0}: '
                               'Unknown exception: {1}'
                               .format(self.plateifu, ee))

        if not self._rss_db:
            raise MarvinError('Could not retrieve RSS for plate-ifu {0}: '
                              'Unknown error.'.format(self.plateifu))

        self.data_origin = 'db'

    def _getRSSFromAPI(self):
        """Initialises the RSS object using the remote API."""

        # Checks that the RSS exists.
        routeparams = {'name': self.plateifu}
        url = marvin.config.urlmap['api']['getRSS']['url'].format(**routeparams)

        # Make the API call
        api.Interaction(url)

    def _initFibers(self):
        """Initialises the object as a list of RSSFiber instances."""

        if self.data_origin == 'file':
            _fibers = [RSSFiber._initFromHDU(hdulist=self._hdu, index=ii)
                       for ii in range(self._hdu[1].data.shape[0])]

        elif self.data_origin == 'db':
            _fibers = [RSSFiber._initFromDB(rssfiber=rssfiber)
                       for rssfiber in self._rss_db]

        else:
            # Makes a call to the API to retrieve all the arrays for all the fibres.

            routeparams = {'name': self.plateifu}
            url = marvin.config.urlmap['api']['getRSSAllFibers']['url'].format(**routeparams)

            # Make the API call
            response = api.Interaction(url)
            data = response.getData()

            wavelength = np.array(data['wavelength'], np.float)

            _fibers = []
            for ii in range(len(data) - 1):
                flux = np.array(data[str(ii)][0], np.float)
                ivar = np.array(data[str(ii)][1], np.float)
                mask = np.array(data[str(ii)][2], np.float)
                _fibers.append(
                    RSSFiber(flux, ivar=ivar, mask=mask, wavelength=wavelength,
                             mangaid=self.mangaid, plateifu=self.plateifu, data_origin='api'))

        list.__init__(self, _fibers)


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

        Spectrum.__init__(self, *args, **kwargs)

    def __repr__(self):
        """Representation for RSSFiber."""

        return ('<Marvin RSSFiber (mangaid={self.mangaid!r}, plateifu={self.plateifu!r}, '
                'data_origin={self.data_origin!r})>'.format(self=self))

    @classmethod
    def _initFromHDU(cls, hdulist, index):
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
    def _initFromDB(cls, rssfiber):
        """Initialites a RSS fiber from the DB."""

        mangaid = rssfiber.cube.mangaid
        plateifu = '{0}-{1}'.format(rssfiber.cube.plate, rssfiber.cube.ifu.name)

        obj = RSSFiber(rssfiber.flux, ivar=rssfiber.ivar, mask=rssfiber.mask,
                       wavelength=rssfiber.cube.wavelength.wavelength, mangaid=mangaid,
                       plateifu=plateifu, data_origin='db')

        return obj
