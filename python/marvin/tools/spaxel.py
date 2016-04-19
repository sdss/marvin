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
from marvin.tools.core import MarvinToolsClass
from marvin.tools.core import MarvinError
from marvin.tools.spectrum import Spectrum
from astropy.io import fits
from marvin.api import api
import marvin
import numpy as np


class Spaxel(MarvinToolsClass, Spectrum):
    """A class to interface with a spaxel in a cube.

    This class represents a fully reduced spaxel, initialised either
    from a file, a database, or remotely via the Marvin API. An spaxel contains
    flux, ivar, and mask information.

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
        drpall (str):
            The path to the drpall file to use. Defaults to
            ``marvin.config.drpall``.
        drpver (str):
            The DRP version to use. Defaults to ``marvin.config.drpver``.

    Return:
        rss:
            An object representing the Spaxel entity.

    """

    def __init__(self, *args, **kwargs):

        self.data_origin = None
        self._hduList = None
        self._spaxel_db = None

        if len(args) > 0:
            self.x = args[0]
            self.y = args[1]
        else:
            self.x = kwargs.pop('x', None)
            self.y = kwargs.pop('y', None)

        assert self.x is not None and self.y is not None

        MarvinToolsClass.__init__(self, *args, **kwargs)

        if self.mode == 'local':
            if self.filename:
                self._getSpaxelFromFile()
            else:
                self._getSpaxelFromDB()
        else:
            self._getSpaxelFromAPI()

        self._createSpectrum()

    def __repr__(self):
        """Spaxel representation."""

        return '<Marvin Spaxel (x={0:d}, y={1:d})>'.format(self.x, self.y)

    def _getFullPath(self, **kwargs):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return MarvinToolsClass._getFullPath(self, 'mangacube', ifu=ifu,
                                             drpver=self._drpver,
                                             plate=plate)

    def _createSpectrum(self):
        """Initialises the :class:`.Spectrum` object based on this Spaxel."""

        if self.data_origin == 'file':
            data = self._hduList['FLUX'].data
            ivar = self._hduList['IVAR'].data
            mask = self._hduList['MASK'].data
            wavelength = self._hduList['WAVE'].data

        elif self.data_origin == 'db':
            data = self._spaxel_db.flux
            ivar = self._spaxel_db.ivar
            mask = self._spaxel_db.mask
            wavelength = self._spaxel_db.cube.wavelength.wavelength

        elif self.data_origin == 'api':
            data = np.array(self._arrays['data'])
            ivar = np.array(self._arrays['ivar'])
            mask = np.array(self._arrays['mask'])
            wavelength = np.array(self._arrays['wavelength'])

        Spectrum.__init__(self, data, ivar=ivar, mask=mask, wavelength=wavelength)

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
                        newHDU = fits.ImageHDU(data=spectrum,
                                               header=ext.header)
                        self._hduList.append(newHDU)

            self.data_origin = 'file'

        except Exception as ee:
            raise MarvinError('Could not initialize via filename: {0}'
                              .format(ee))

    def _getSpaxelFromDB(self):
        """Initialises the spaxel object from the DB."""

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

        self.data_origin = 'db'

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

        # If the response is valid, the spaxel exists.
        self.data_origin = 'api'

        return response

    @classmethod
    def _initFromData(cls, x, y, data):
        """Initialises a spaxel from and HDUList or DB object."""

        obj = Spaxel.__new__(Spaxel)

        obj.data_origin = None
        obj._hduList = None
        obj._spaxel_db = None
        obj.filename = None

        obj.x = x
        obj.y = y

        if isinstance(data, fits.HDUList):
            obj._getSpaxelFromFile(cubeHDU=data)
        elif hasattr(data, '__tablename__') and data.__tablename__ == 'spaxel':
            obj._getSpaxelFromDB()
        else:
            raise MarvinError('cannot initialise a Spaxel from data type {0}'
                              .format(type(data)))

        obj._createSpectrum()

        return obj
