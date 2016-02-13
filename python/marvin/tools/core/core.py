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
from marvin import config, session, datadb
from marvin.tools.core import MarvinUserWarning, MarvinError
from marvin.utils.db import testDbConnection
import warnings
import os

try:
    from sdss_access.path import Path
except ImportError:
    Path = None


__all__ = ['MarvinToolsClass']


mangaid_to_plateifu = {'1-209232': '8485-1901', '12-193534': '7443-3701'}


class MarvinToolsClass(object):

    def __init__(self, *args, **kwargs):
        """Marvin tools main super class.

        This super class implements the decision tree for using local files,
        database, or remote connection when initialising a Marvin tools
        object.

        """

        self.filename = kwargs.get('filename', None)
        self.mangaid = kwargs.get('mangaid', None)
        self.plateifu = kwargs.get('plateifu', None)
        self.mode = kwargs.get('mode', None)

        if self.mode is None:
            self.mode = config.mode

        args = [self.filename, self.plateifu, self.mangaid]
        errmsg = 'Enter filename, plateifu, or mangaid!'
        assert any(args), errmsg
        assert sum([bool(arg) for arg in args]) == 1, errmsg

        if self.mangaid:
            if self.mangaid in mangaid_to_plateifu:
                self.plateifu = mangaid_to_plateifu[self.mangaid]
            else:
                raise MarvinError('mangaid={0} not found in the dictionary'
                                  .format(self.mangaid))

        if self.mode == 'local':
            self._doLocal()
        elif self.mode == 'remote':
            self._doRemote()
        elif self.mode == 'auto':
            try:
                self._doLocal()
            except:
                warnings.warn('local mode failed. Trying remote now.',
                              MarvinUserWarning)
                self._doRemote()

    def _doLocal(self):
        """Tests if it's possible to load the data locally."""

        if self.filename:

            if os.path.exists(self.filename):
                self.mode = 'local'
            else:
                raise MarvinError('input file {0} not found'
                                  .format(self.filename))

        elif self.plateifu:

            dbStatus = testDbConnection(session)
            print(dbStatus)

            if dbStatus['good']:
                self.mode = 'local'
            else:
                warnings.warn(
                    'DB connection failed with error: {0}.'
                    .format(dbStatus['error']), MarvinUserWarning)

            fullpath = self._getFullPath()

            if fullpath and os.path.exists(fullpath):
                self.mode = 'local'
                self.filename = fullpath
            else:
                if config.download:
                    raise NotImplementedError('sdsssync not yet implemented')
                    # When implemented, this should download the data and
                    # then kwargs['filename'] = downloaded_path and
                    # kwargs['mode'] = local
                else:
                    raise MarvinError('this is the end of the road. Try '
                                      'using some reasonable inputs.')

    def _doRemote(self):
        """Tests if remote connection is possible."""

        if self.filename:
            raise MarvinError('filename not allowed in remote mode.')
        else:
            self.mode = 'remote'

    def _getFullPath(self, pathType, **pathParams):
        """Returns the full path of the file in the tree."""

        if not Path:
            raise MarvinError('sdss_access is not installed')
        else:
            try:
                fullpath = Path().full(pathType, **pathParams)
            except Exception as ee:
                warnings.warn(
                    'sdss_access was not able to retrieve the full path of '
                    'the file. Error message is: {0}'.format(str(ee)),
                    MarvinUserWarning)
                fullpath = None

        return fullpath
