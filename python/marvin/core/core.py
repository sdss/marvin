#!/usr/bin/env python
# encoding: utf-8
"""

core.py

Licensed under a 3-clause BSD license.

Revision history:
    12 Feb 2016 J. Sánchez-Gallego
      Initial version

"""

from __future__ import division
from __future__ import print_function
import marvin
from marvin.core import MarvinUserWarning, MarvinError, MarvinMissingDependency
from marvin.utils.general import mangaid2plateifu
from marvin.utils.db import testDbConnection
import warnings
import os

try:
    from sdss_access.path import Path
except ImportError:
    Path = None

try:
    from sdss_access import RsyncAccess
except ImportError:
    RsyncAccess = None

__all__ = ['MarvinToolsClass']


def kwargsGet(kwargs, key, replacement):
    """As kwargs.get but handles uses replacemente if the value is None."""

    if key not in kwargs:
        return replacement
    elif key in kwargs and kwargs[key] is None:
        return replacement
    else:
        return kwargs[key]


class MarvinToolsClass(object):

    def __init__(self, *args, **kwargs):
        """Marvin tools main super class.

        This super class implements the decision tree for using local files,
        database, or remote connection when initialising a Marvin tools
        object.

        """

        self.filename = kwargsGet(kwargs, 'filename', None)
        self.mangaid = kwargsGet(kwargs, 'mangaid', None)
        self.plateifu = kwargsGet(kwargs, 'plateifu', None)
        self.mode = kwargsGet(kwargs, 'mode', marvin.config.mode)
        self._drpall = kwargsGet(kwargs, 'drpall', marvin.config.drpall)
        self._drpver = kwargsGet(kwargs, 'drpver', marvin.config.drpver)
        self._dapver = kwargsGet(kwargs, 'dapver', marvin.config.dapver)
        self._forcedownload = kwargsGet(kwargs, 'download', marvin.config.download)
        self._dapver = kwargsGet(kwargs, 'drpver', marvin.config.dapver)
        self.data_origin = None

        args = [self.filename, self.plateifu, self.mangaid]
        errmsg = 'Enter filename, plateifu, or mangaid!'
        assert any(args), errmsg
        assert sum([bool(arg) for arg in args]) == 1, errmsg

        if self.mangaid:
            self.plateifu = mangaid2plateifu(self.mangaid, drpall=self._drpall, drpver=self._drpver)

        if self.mode == 'local':
            self._doLocal()
        elif self.mode == 'remote':
            self._doRemote()
        elif self.mode == 'auto':
            try:
                self._doLocal()
            except:
                warnings.warn('local mode failed. Trying remote now.', MarvinUserWarning)
                self._doRemote()

        # Sanity check to make sure data_origin has been properly set.
        assert self.data_origin in ['file', 'db', 'api'], 'data_origin is not properly set.'

    def _doLocal(self):
        """Tests if it's possible to load the data locally."""

        if self.filename:

            if os.path.exists(self.filename):
                self.mode = 'local'
                self.data_origin = 'file'
            else:
                raise MarvinError('input file {0} not found'.format(self.filename))

        elif self.plateifu:

            dbStatus = testDbConnection(marvin.marvindb.session)

            if dbStatus['good']:
                self.mode = 'local'
                self.data_origin = 'db'
            else:
                warnings.warn('DB connection failed with error: {0}.'.format(dbStatus['error']),
                              MarvinUserWarning)

                fullpath = self._getFullPath()

                if fullpath and os.path.exists(fullpath):
                    self.mode = 'local'
                    self.filename = fullpath
                    self.data_origin = 'file'
                else:
                    if self._forcedownload:
                        self.download()
                        # raise NotImplementedError('sdsssync not yet implemented')
                        self.data_origin = 'file'
                        # When implemented, this should download the data and
                        # then kwargs['filename'] = downloaded_path and
                        # kwargs['mode'] = local
                    else:
                        raise MarvinError('this is the end of the road. '
                                          'Try using some reasonable inputs.')

    def _doRemote(self):
        """Tests if remote connection is possible."""

        if self.filename:
            raise MarvinError('filename not allowed in remote mode.')
        else:
            self.mode = 'remote'
            self.data_origin = 'api'

    def download(self, pathType, url=None, **pathParams):
        ''' Download using sdss_access Rsync '''
        if not RsyncAccess:
            raise MarvinError('sdss_access is not installed')
        else:
            rsync_access = RsyncAccess()
            rsync_access.remote()
            rsync_access.add(pathType, **pathParams)
            rsync_access.set_stream()
            rsync_access.commit()
            paths = rsync_access.get_paths()
            self.filename = paths[0]  # doing this for single files, may need to change

    def _getFullPath(self, pathType, url=None, **pathParams):
        """Returns the full path of the file in the tree."""

        if not Path:
            raise MarvinMissingDependency('sdss_access is not installed')
        else:
            try:
                if url:
                    fullpath = Path().url(pathType, **pathParams)
                else:
                    fullpath = Path().full(pathType, **pathParams)
            except Exception as ee:
                warnings.warn('sdss_access was not able to retrieve the full path of the file. '
                              'Error message is: {0}'.format(str(ee)), MarvinUserWarning)
                fullpath = None

        return fullpath


class Dotable(dict):
    """A custom dict class that allows dot access to nested dictionaries.

    Copied from http://hayd.github.io/2013/dotable-dictionaries/. Note that
    this allows you to use dots to get dictionary values, but not to set them.

    """

    __getattr__ = dict.__getitem__

    def __init__(self, d):
        self.update(**dict((k, self.parse(v)) for k, v in d.iteritems()))

    @classmethod
    def parse(cls, v):
        if isinstance(v, dict):
            return cls(v)
        elif isinstance(v, list):
            return [cls.parse(i) for i in v]
        else:
            return v


class DotableCaseInsensitive(Dotable):
    """Like dotable but access to attributes and keys is case insensitive."""

    def _match(self, list_of_keys, value):

        lower_values = map(str.lower, list_of_keys)
        if value.lower() in lower_values:
            return list_of_keys[lower_values.index(value.lower())]
        else:
            return False

    def __getattr__(self, value):
        return self.__getitem__(value)

    def __getitem__(self, value):
        key = self._match(self.keys(), value)
        if key is False:
            raise KeyError('{0} key or attribute not found'.format(value))
        return dict.__getitem__(self, key)
