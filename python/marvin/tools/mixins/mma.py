# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-07-28 17:26:41
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last Modified time: 2018-08-27 13:57:09

from __future__ import absolute_import, division, print_function

import abc
import os
import re
import time
import warnings

import six

from marvin import config, log
from marvin.core.exceptions import MarvinError, MarvinMissingDependency, MarvinUserWarning
from marvin.utils.db import testDbConnection
from marvin.utils.general.general import mangaid2plateifu


try:
    from sdss_access.path import Path
except ImportError:
    Path = None

try:
    from sdss_access import RsyncAccess
except ImportError:
    RsyncAccess = None

__all__ = ['MMAMixIn']


class MMAMixIn(object, six.with_metaclass(abc.ABCMeta)):
    """A mixin that provides multi-modal data access.

    Add this mixin to any new class object to provide that class with
    the Multi-Modal Data Access System for using local files, database,
    or remote connection when initializing new objects.  See :ref:`decision tree <marvin-dma>`

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
            based on the ``release``
        download (bool):
            If ``True``, the data will be downloaded on instantiation. See
            :ref:`marvin-download-objects`.
        ignore_db (bool):
            If ``True``, the local data-origin `db` will be ignored.

    Attributes:
        data (:class:`~astropy.io.fits.HDUList`, SQLAlchemy object, or dict):
            Depending on the access mode, ``data`` is populated with the
            |HDUList| from the FITS file, a
            `SQLAlchemy <http://www.sqlalchemy.org>`_ object, or a dictionary
            of values returned by an API call.
        data_origin ({'file', 'db', 'api'}):
            Indicates the origin of the data, either from a file, the DB, or
            an API call.
        filename (str):
            The path of the file used, if any.
        mangaid (str):
            The mangaid of the target.
        plateifu:
            The plateifu of the target
        release:
            The data release

    """

    def __init__(self, input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None, drpall=None, download=None,
                 ignore_db=False):
        self.data = data
        self.data_origin = None
        self._ignore_db = ignore_db

        self.filename = filename
        self.mangaid = mangaid
        self.plateifu = plateifu

        self.mode = mode if mode is not None else config.mode

        self._release = release if release is not None else config.release

        self._drpver, self._dapver = config.lookUpVersions(release=self._release)
        self._drpall = config._getDrpAllPath(self._drpver) if drpall is None else drpall

        self._forcedownload = download if download is not None else config.download

        self._determine_inputs(input)

        assert self.mode in ['auto', 'local', 'remote']
        assert self.filename is not None or self.plateifu is not None, 'no inputs set.'

        self.datamodel = None
        self._set_datamodel()

        if self.mode == 'local':
            self._doLocal()
        elif self.mode == 'remote':
            self._doRemote()
        elif self.mode == 'auto':
            try:
                self._doLocal()
            except Exception as ee:

                if self.filename:
                    # If the input contains a filename we don't want to go into remote mode.
                    raise(ee)
                else:
                    log.debug('local mode failed. Trying remote now.')
                    self._doRemote()

        # Sanity check to make sure data_origin has been properly set.
        assert self.data_origin in ['file', 'db', 'api'], 'data_origin is not properly set.'

    def _determine_inputs(self, input):
        """Determines what inputs to use in the decision tree."""

        if input:

            assert self.filename is None and self.plateifu is None and self.mangaid is None, \
                'if input is set, filename, plateifu, and mangaid cannot be set.'

            assert isinstance(input, six.string_types), 'input must be a string.'

            input_dict = self._parse_input(input)

            if input_dict['plate'] is not None and input_dict['ifu'] is not None:
                self.plateifu = input
            elif input_dict['plate'] is not None and input_dict['ifu'] is None:
                self._plate = input
            elif input_dict['mangaid'] is not None:
                self.mangaid = input
            else:
                # Assumes the input must be a filename
                self.filename = input

        if self.filename is None and self.mangaid is None and self.plateifu is None:
            raise MarvinError('no inputs defined.')

        if self.filename:
            self.mangaid = None
            self.plateifu = None

            if self.mode == 'remote':
                raise MarvinError('filename not allowed in remote mode.')

            assert os.path.exists(self.filename), \
                'filename {} does not exist.'.format(str(self.filename))

        elif self.plateifu:
            assert not self.filename, 'invalid set of inputs.'

        elif self.mangaid:
            assert not self.filename, 'invalid set of inputs.'
            self.plateifu = mangaid2plateifu(self.mangaid,
                                             drpall=self._drpall,
                                             drpver=self._drpver)

        elif self._plate:
            assert not self.filename, 'invalid set of inputs.'

    @staticmethod
    def _parse_input(value):
        """Parses and input and determines plate, ifu, and mangaid."""

        # Number of IFUs per size
        n_ifus = {19: 2, 37: 4, 61: 4, 91: 2, 127: 5, 7: 12}

        return_dict = {'plate': None, 'ifu': None, 'mangaid': None}

        plateifu_pattern = re.compile(r'([0-9]{4,5})-([0-9]{4,9})')
        ifu_pattern = re.compile('(7|127|[0-9]{2})([0-9]{2})')
        mangaid_pattern = re.compile(r'[0-9]{1,3}-[0-9]+')
        plateid_pattern = re.compile('([0-9]{4,})(?!-)(?<!-)')

        plateid_match = re.match(plateid_pattern, value)
        plateifu_match = re.match(plateifu_pattern, value)
        mangaid_match = re.match(mangaid_pattern, value)

        # Check whether the input value matches the plateifu pattern
        if plateifu_match is not None:
            plate, ifu = plateifu_match.groups(0)

            # If the value matches a plateifu, checks that the ifu is a valid one.
            ifu_match = re.match(ifu_pattern, ifu)
            if ifu_match is not None:
                ifu_size, ifu_id = map(int, ifu_match.groups(0))
                if ifu_id <= n_ifus[ifu_size]:
                    return_dict['plate'] = plate
                    return_dict['ifu'] = ifu

        # Check whether this is a mangaid
        elif mangaid_match is not None:
            return_dict['mangaid'] = value

        # Check whether this is a plate
        elif plateid_match is not None:
            return_dict['plate'] = value

        return return_dict

    @staticmethod
    def _get_ifus(minis=None):
        ''' Returns a list of all the allowed IFU designs ids

        Parameters:
            minis (bool):
                If True, includes the mini-bundles

        Returns:
            A list of IFU designs

        '''

        # Number of IFUs per size
        n_ifus = {19: 2, 37: 4, 61: 4, 91: 2, 127: 5, 7: 12}

        # Pop the minis
        if not minis:
            __ = n_ifus.pop(7)

        ifus = ['{0}{1:02d}'.format(key, i + 1) for key, value in n_ifus.items() for i in range(value)]
        return ifus

    def _set_datamodel(self):
        """Sets the datamodel for this object. Must be overridden by each subclass."""
        pass

    def _doLocal(self):
        """Tests if it's possible to load the data locally."""

        if self.filename:

            if os.path.exists(self.filename):
                self.mode = 'local'
                self.data_origin = 'file'
            else:
                raise MarvinError('input file {0} not found'.format(self.filename))

        elif self.plateifu:

            from marvin import marvindb
            if marvindb:
                testDbConnection(marvindb.session)

            if marvindb.db and not self._ignore_db:
                self.mode = 'local'
                self.data_origin = 'db'
            else:
                fullpath = self._getFullPath()

                if fullpath and os.path.exists(fullpath):
                    self.mode = 'local'
                    self.filename = fullpath
                    self.data_origin = 'file'
                else:
                    if self._forcedownload:
                        self.download()
                        self.data_origin = 'file'
                    else:
                        raise MarvinError('failed to retrieve data using '
                                          'input parameters.')

    def _doRemote(self):
        """Tests if remote connection is possible."""

        if self.filename:
            raise MarvinError('filename not allowed in remote mode.')
        else:
            self.mode = 'remote'
            self.data_origin = 'api'

    def download(self, pathType=None, **pathParams):
        """Download using sdss_access Rsync"""

        # check for public release
        is_public = 'DR' in self._release
        rsync_release = self._release.lower() if is_public else None

        if not RsyncAccess:
            raise MarvinError('sdss_access is not installed')
        else:
            rsync_access = RsyncAccess(public=is_public, release=rsync_release)
            rsync_access.remote()
            rsync_access.add(pathType, **pathParams)
            rsync_access.set_stream()
            rsync_access.commit()
            paths = rsync_access.get_paths()
            # adding a millisecond pause for download to finish and file existence to register
            time.sleep(0.001)

            self.filename = paths[0]  # doing this for single files, may need to change

    @abc.abstractmethod
    def _getFullPath(self, pathType=None, url=None, **pathParams):
        """Returns the full path of the file in the tree.

        This method must be overridden by each subclass.

        """

        # check for public release
        is_public = 'DR' in self._release
        path_release = self._release.lower() if is_public else None

        if not Path:
            raise MarvinMissingDependency('sdss_access is not installed')
        else:
            path = Path(public=is_public, release=path_release)
            try:
                if url:
                    fullpath = path.url(pathType, **pathParams)
                else:
                    fullpath = path.full(pathType, **pathParams)
            except Exception as ee:
                warnings.warn('sdss_access was not able to retrieve the full path of the file. '
                              'Error message is: {0}'.format(str(ee)), MarvinUserWarning)
                fullpath = None

        return fullpath

    @property
    def release(self):
        """Returns the release."""

        return self._release

    @release.setter
    def release(self, value):
        """Fails when trying to set the release after instantiation."""

        raise MarvinError('the release cannot be changed once the object has been instantiated.')

    @property
    def plate(self):
        """Returns the plate id."""

        return int(self.plateifu.split('-')[0])

    @property
    def ifu(self):
        """Returns the IFU."""

        return int(self.plateifu.split('-')[1])
