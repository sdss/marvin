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

import os
import warnings

import astropy.io.fits

from brain.core.exceptions import BrainError

import marvin
import marvin.api.api
from marvin.core import marvin_pickle

from marvin.core.exceptions import MarvinUserWarning, MarvinError
from marvin.core.exceptions import MarvinMissingDependency, MarvinBreadCrumb
from marvin.utils.db import testDbConnection
from marvin.utils.general import mangaid2plateifu, get_nsa_data

try:
    from sdss_access.path import Path
except ImportError:
    Path = None

try:
    from sdss_access import RsyncAccess
except ImportError:
    RsyncAccess = None


__all__ = ['MarvinToolsClass', 'Dotable', 'DotableCaseInsensitive']


def kwargsGet(kwargs, key, replacement):
    """As kwargs.get but uses replacement if the value is None."""

    if key not in kwargs:
        return replacement
    elif key in kwargs and kwargs[key] is None:
        return replacement
    else:
        return kwargs[key]


breadcrumb = MarvinBreadCrumb()


class MarvinToolsClass(object):

    def __init__(self, *args, **kwargs):
        """Marvin tools main super class.

        This super class implements the decision tree for using local files,
        database, or remote connection when initialising a Marvin tools
        object.

        """

        self.data = kwargsGet(kwargs, 'data', None)

        self.filename = kwargsGet(kwargs, 'filename', None)
        if self.filename:
            self.filename = os.path.realpath(os.path.expanduser(self.filename))

        self.mangaid = kwargsGet(kwargs, 'mangaid', None)
        self.plateifu = kwargsGet(kwargs, 'plateifu', None)

        self.mode = kwargsGet(kwargs, 'mode', marvin.config.mode)

        self._release = kwargsGet(kwargs, 'release', marvin.config.release)

        self._drpver, self._dapver = marvin.config.lookUpVersions(release=self._release)
        self._drpall = kwargsGet(kwargs, 'drpall', marvin.config._getDrpAllPath(self._drpver))

        self._nsa = None
        self.nsa_source = kwargs.pop('nsa_source', 'auto')
        assert self.nsa_source in ['auto', 'nsa', 'drpall'], \
            'nsa_source must be one of auto, nsa, or drpall'

        self._forcedownload = kwargsGet(kwargs, 'download', marvin.config.download)

        self.data_origin = None

        args = [self.filename, self.plateifu, self.mangaid]
        assert any(args), 'Enter filename, plateifu, or mangaid!'

        if self.filename:
            self.plateifu = None
            self.mangaid = None
        elif self.plateifu:
            self.filename = None
            self.mangaid = None
        elif self.mangaid:
            self.filename = None
            self.plateifu = mangaid2plateifu(self.mangaid,
                                             drpall=self._drpall,
                                             drpver=self._drpver)

        # drop breadcrumb
        breadcrumb.drop(message='Initializing MarvinTool {0}'.format(self.__class__),
                        category=self.__class__)

        if self.mode == 'local':
            self._doLocal()
        elif self.mode == 'remote':
            self._doRemote()
        elif self.mode == 'auto':
            try:
                self._doLocal()
            except Exception:
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

            if marvin.marvindb.db:
                self.mode = 'local'
                self.data_origin = 'db'
            else:
                # TODO - fix verbosity later, check for more advanced db failures
                #warnings.warn('DB connection failed with error: {0}.'.format(dbStatus['error']),
                #              MarvinUserWarning)

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

    def download(self, pathType=None, **pathParams):
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

    def _getFullPath(self, pathType=None, url=None, **pathParams):
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

    def ToolInteraction(self, url, params=None):
        """Runs an Interaction and passes self._release."""

        params = params or {'release': self._release}
        return marvin.api.api.Interaction(url, params=params)

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

        return marvin_pickle.restore(path, delete=delete)

    @property
    def nsa(self):
        """Returns the contents of the NSA catalogue for this target."""

        if self._nsa is None:

            if self.nsa_source == 'auto':
                if self.data_origin == 'file':
                    nsa_source = 'drpall'
                else:
                    nsa_source = 'nsa'
            else:
                nsa_source = self.nsa_source

            try:
                self._nsa = get_nsa_data(self.mangaid, mode='auto',
                                         source=nsa_source,
                                         drpver=self._drpver,
                                         drpall=self._drpall)
            except (MarvinError, BrainError):
                warnings.warn('cannot load NSA information for mangaid={0}.'.format(self.mangaid))
                return None

        return self._nsa

    @property
    def release(self):
        """Returns the release."""

        return self._release

    @release.setter
    def release(self, value):
        """Fails when trying to set the release after instatiation."""

        raise MarvinError('the release cannot be changed once the object has been instantiated.')


class Dotable(dict):
    """A custom dict class that allows dot access to nested dictionaries.

    Copied from http://hayd.github.io/2013/dotable-dictionaries/. Note that
    this allows you to use dots to get dictionary values, but not to set them.

    """

    def __getattr__(self, value):
        if '__' in value:
            return dict.__getattr__(self, value)
        else:
            return self.__getitem__(value)

    # def __init__(self, d):
    #     dict.__init__(self, ((k, self.parse(v)) for k, v in d.iteritems()))

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

        lower_values = [str(xx).lower() for xx in list_of_keys]
        if value.lower() in lower_values:
            return list_of_keys[lower_values.index(value.lower())]
        else:
            return False

    def __getattr__(self, value):
        if '__' in value:
            return super(DotableCaseInsensitive, self).__getattr__(value)
        return self.__getitem__(value)

    def __getitem__(self, value):
        key = self._match(list(self.keys()), value)
        if key is False:
            raise KeyError('{0} key or attribute not found'.format(value))
        return dict.__getitem__(self, key)
