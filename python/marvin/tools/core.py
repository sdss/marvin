#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Filename: core.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-09 09:55:04

from __future__ import absolute_import, division, print_function

import abc
import warnings

import astropy.io.fits

import marvin
import marvin.api.api
from marvin.core import marvin_pickle
from marvin.core.exceptions import MarvinBreadCrumb, MarvinError, MarvinUserWarning
from marvin.tools.mixins import MMAMixIn
from marvin.utils.general.maskbit import get_manga_target


try:
    from sdss_access.path import Path
except ImportError:
    Path = None

try:
    from sdss_access import RsyncAccess
except ImportError:
    RsyncAccess = None


__ALL__ = ['MarvinToolsClass']


def kwargsGet(kwargs, key, replacement):
    """As kwargs.get but uses replacement if the value is None."""

    if key not in kwargs:
        return replacement
    elif key in kwargs and kwargs[key] is None:
        return replacement
    else:
        return kwargs[key]


breadcrumb = MarvinBreadCrumb()


class MarvinToolsClass(MMAMixIn):
    """Marvin tools main base class.

    This super class implements the :ref:`decision tree <marvin-dma>`
    for using local files, database, or remote connection when
    initialising a Marvin tools object.

    Parameters:
        input (str):
            A string that can be a filename, plate-ifu, or mangaid. It will be
            automatically identified based on its unique format. This argument
            is always the first one, so it can be defined without the keyword
            for convenience.
        filename (str):
            The path of the file containing the file to load. If set,
            ``input`` is ignored.
        mangaid (str):
            The mangaid of the file to load. If set, ``input`` is ignored.
        plateifu (str):
            The plate-ifu of the data cube to load. If set, ``input`` is
            ignored.
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See :ref:`mode-decision-tree`.
        data (:class:`~astropy.io.fits.HDUList`, SQLAlchemy object, or None):
            An astropy ``HDUList`` or a SQLAlchemy object, to be used for
            initialisation. If ``None``, the :ref:`normal <marvin-dma>`` mode
            will be used.
        release (str):
            The MPL/DR version of the data to use.
        drpall (str):
            The path to the
            `drpall <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/metadata#DRP:DRPall>`_
            file to use. If not set it will use the default path for the file
            based on the ``release``.
        download (bool):
            If ``True``, the data will be downloaded on instantiation. See
            :ref:`marvin-download-objects`.

    Attributes:
        data (:class:`~astropy.io.fits.HDUList`, SQLAlchemy object, or dict):
            Depending on the access mode, ``data`` is populated with the
            |HDUList| from the FITS file, a
            `SQLAlchemy <http://www.sqlalchemy.org>`_ object, or a dictionary
            of values returned by an API call.
        datamodel:
            A datamodel object, whose type depends on the subclass that
            initialises the datamodel.
        data_origin ({'file', 'db', 'api'}):
            Indicates the origin of the data, either from a file, the DB, or
            an API call.
        filename (str):
            The path of the file used, if any.
        mangaid (str):
            The mangaid of the target.
        plateifu:
            The plateifu of the target

    """

    def __init__(self, input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None, drpall=None, download=None):

        MMAMixIn.__init__(self, input=input, filename=filename, mangaid=mangaid,
                          plateifu=plateifu, mode=mode, data=data, release=release,
                          download=download)

        # drop breadcrumb
        breadcrumb.drop(message='Initializing MarvinTool {0}'.format(self.__class__),
                        category=self.__class__)

        # Load VACs
        from marvin.contrib.vacs.base import VACMixIn
        self.vacs = VACMixIn.get_vacs(self)

    def _toolInteraction(self, url, params=None):
        """Runs an Interaction and passes self._release."""

        params = params or {'release': self._release}
        return marvin.api.api.Interaction(url, params=params)

    @staticmethod
    def _check_file(header, data, objtype):
        ''' Check the file input to ensure correct tool '''

        # get/check various header keywords
        bininhdr = ('binkey' in header) or ('bintype' in header)
        dapinhdr = 'dapfrmt' in header
        dapfrmt = header['DAPFRMT'] if dapinhdr else None

        # check the file
        if objtype == 'Maps' or objtype == 'ModelCube':
            # get indices in daptype
            daptype = ['MAPS', 'LOGCUBE']
            dapindex = daptype.index("MAPS") if objtype == 'Maps' else daptype.index("LOGCUBE")
            altdap = 1 - dapindex

            # check for emline_gflux extension
            gfluxindata = 'EMLINE_GFLUX' in data
            wronggflux = (gfluxindata and objtype == 'ModelCube') or \
                         (not gfluxindata and objtype == 'Maps')

            if not bininhdr:
                raise MarvinError('Trying to open a non DAP file with Marvin {0}'.format(objtype))
            else:
                if (dapfrmt and dapfrmt != daptype[dapindex]) or (wronggflux):
                    raise MarvinError('Trying to open a DAP {0} with Marvin {1}'
                                      .format(daptype[altdap], objtype))
        elif objtype == 'Cube':
            if bininhdr or dapinhdr:
                raise MarvinError('Trying to open a DAP file with Marvin Cube')

    def __getstate__(self):

        if self.data_origin == 'db':
            raise MarvinError('objects with data_origin=\'db\' cannot be saved.')

        odict = self.__dict__.copy()
        del odict['data']

        return odict

    def __setstate__(self, idict):

        data = None
        if idict['data_origin'] == 'file':
            try:
                data = astropy.io.fits.open(idict['filename'])
            except Exception as ee:
                warnings.warn('there was a problem reloading the FITS object: {0}. '
                              'The object has been unpickled but not all the functionality '
                              'will be available.'.format(str(ee)), MarvinUserWarning)

        self.__dict__.update(idict)
        self.data = data

    def save(self, path=None, overwrite=False):
        """Pickles the object.

        If ``path=None``, uses the default location of the file in the tree
        but changes the extension of the file to ``.mpf``. Returns the path
        of the saved pickle file.

        Parameters:
            obj:
                Marvin object to pickle.
            path (str):
                Path of saved file. Default is ``None``.
            overwrite (bool):
                If ``True``, overwrite existing file. Default is ``False``.

        Returns:
            str:
                Path of saved file.

        """

        return marvin_pickle.save(self, path=path, overwrite=overwrite)

    @classmethod
    def restore(cls, path, delete=False):
        """Restores a MarvinToolsClass object from a pickled file.

        If ``delete=True``, the pickled file will be removed after it has been
        unplickled. Note that, for objects with ``data_origin='file'``, the
        original file must exists and be in the same path as when the object
        was first created.

        """

        from marvin.contrib.vacs.base import VACMixIn

        res = marvin_pickle.restore(path, delete=delete)
        res.vacs = VACMixIn.get_vacs(res)

        return res

    @property
    def source_is_fits_hdulist_file(self):
        source_is_file = (self.data_origin == 'file')
        data_from_hdulist = isinstance(self.data, astropy.io.fits.HDUList)

        return source_is_file and data_from_hdulist

    def close(self):
        if self.source_is_fits_hdulist:
            try:
                self.data.close()
            except Exception as ee:
                warnings.warn('failed to close FITS instance: {0}'.format(ee), MarvinUserWarning)
        else:
            warnings.warn(
                'data-origin {0} ({1}) not closeable'.format(self.data_origin, self.data.__class__),
                MarvinUserWarning)

    def __del__(self):
        """Destructor for closing FITS files."""

        self.close()

    def __enter__(self):
        if not self.source_is_fits_hdulist_file:
            raise MarvinError('to use Tools as a context-manager, self.data_origin must be \'file\', '
                              'and self.data must be a FITS HDUList')

    def __exit__(self, type, value, traceback):
        self.close()

        return True

    @property
    def quality_flag(self):
        """Return quality flag."""

        if self.datamodel.qual_flag is None:
            return None

        try:
            dapqual = self._bitmasks['MANGA_' + self.datamodel.qual_flag]
        except KeyError:
            warnings.warn('cannot find bitmask MANGA_{!r}'.format(self.datamodel.qual_flag))
            dapqual = None
        else:
            dapqual.mask = int(self.header[self.datamodel.qual_flag])

        return dapqual

    @property
    def manga_target1(self):
        """Return MANGA_TARGET1 flag."""
        return get_manga_target('1', self._bitmasks, self.header)

    @property
    def manga_target2(self):
        """Return MANGA_TARGET2 flag."""
        return get_manga_target('2', self._bitmasks, self.header)

    @property
    def manga_target3(self):
        """Return MANGA_TARGET3 flag."""
        return get_manga_target('3', self._bitmasks, self.header)

    @property
    def target_flags(self):
        """Bundle MaNGA targeting flags."""
        return [self.manga_target1, self.manga_target2, self.manga_target3]

    def getImage(self):
        ''' Retrieves the Image :class:`~marvin.tools.image.Image` for this object '''

        image = marvin.tools.image.Image(plateifu=self.plateifu, release=self.release)
        return image
