# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2016-09-15 14:50:00
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2016-09-15 16:00:40

from __future__ import print_function, division, absolute_import

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

import marvin
import marvin.core.exceptions
import marvin.tools.spaxel
import marvin.tools.maps
import marvin.utils.general.general

from marvin.api.api import Interaction
from marvin.core import MarvinToolsClass
from marvin.core.exceptions import MarvinError


class ModelCube(MarvinToolsClass):
    """A class to interface with MaNGA DAP model cubes.

    This class represents a fully reduced DAP model cube, initialised either
    from a file, a database, or remotely via the Marvin API.

    Parameters:
        filename (str):
            The path of the file containing the data cube to load.
        mangaid (str):
            The mangaid of the data cube to load.
        plateifu (str):
            The plate-ifu of the data cube to load (either ``mangaid`` or
            ``plateifu`` can be used, but not both).
        binmode (str):
            The binning mode of the model cube to load
        template (str):
            The stellar template type of the model cube to load
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See :doc:`mode-decision-tree>`.
        drpall (str):
            The path to the drpall file to use. Defaults to
            ``marvin.config.drpall``.
        drpver (str):
            The DRP version to use. Defaults to ``marvin.config.drpver``.

    Return:
        cube:
            An object representing the data cube.

    """

    def _getFullPath(self, **kwargs):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')
        daptype = '{0}-{1}'.format(self.binmode, self.template)

        return super(ModelCube, self)._getFullPath('mangadap5', ifu=ifu,
                                                   drpver=self._drpver,
                                                   dapver=self._dapver,
                                                   plate=plate, mode='LOGCUBE',
                                                   daptype=daptype)

    def download(self, **kwargs):
        ''' Downloads the cube using sdss_access - Rsync '''
        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')
        daptype = '{0}-{1}'.format(self.binmode, self.template)

        return super(ModelCube, self).download('mangadap5', ifu=ifu,
                                               drpver=self._drpver,
                                               dapver=self._dapver,
                                               plate=plate, mode='LOGCUBE',
                                               daptype=daptype)

    def __init__(self, *args, **kwargs):

        # TODO: consolidate _hdu/_cube in data. This class needs a clean up.
        # Can use Maps or Spaxel as an example. For now I'm adding more
        # clutter to avoid breaking things (JSG).

        self._hdu = None
        self._cube = None
        self._shape = None

        self.filename = None
        self.wcs = None
        self.data = None
        self.wavelength = None
        self.model = None
        self.redcorr = None

        skip_check = kwargs.get('skip_check', False)

        super(ModelCube, self).__init__(*args, **kwargs)

        if self.data_origin == 'file':
            try:
                self._openFile()
            except IOError as e:
                raise MarvinError('Could not initialize via filename: {0}'.format(e))
            self.plateifu = self.header['PLATEIFU'].strip()

        elif self.data_origin == 'db':
            try:
                self._getCubeFromDB()
            except RuntimeError as e:
                raise MarvinError('Could not initialize via db: {0}'.format(e))

        elif self.data_origin == 'api':
            if not skip_check:
                self._openCubeRemote()

        self.ifu = int(self.header['IFUDSGN'])
        self.ra = float(self.header['OBJRA'])
        self.dec = float(self.header['OBJDEC'])
        self.plate = int(self.header['PLATEID'])
        self.mangaid = self.header['MANGAID']
        self._isbright = 'APOGEE' in self.header['SRVYMODE']
        self.dir3d = 'mastar' if self._isbright else 'stack'

    def __repr__(self):
        """Representation for ModelCube."""

        return ('<Marvin ModelCube (plateifu={0}, mode={1}, data_origin={2})>'
                .format(repr(self.plateifu), repr(self.mode),
                        repr(self.data_origin)))

    def _openFile(self):
        """Initialises a cube from a file."""

        self._useDB = False
        try:
            self._hdu = fits.open(self.filename)
            self.data = self._hdu
        except IOError as err:
            raise IOError('IOError: Filename {0} cannot be found: {1}'.format(self.filename, err))

        self.header = self._hdu['PRIMARY'].header
        self.wcs = WCS(self._hdu['FLUX'].header)
        self.wavelength = self._hdu['WAVE'].data
