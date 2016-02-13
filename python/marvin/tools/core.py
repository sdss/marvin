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
from marvin import config
from marvin.utils.db import testDbConnection
import warnings
import os

try:
    from sdss_access.path import Path
except ImportError:
    Path = None


__all__ = ['MarvinToolsClass']


mangaid_to_plateifu = {'1-209232': '8485-1901', '12-193534': '7443-3701'}


class MarvinError(Exception):
    pass


class MarvinWarning(Warning):
    """Base warning for Marvin."""
    pass


class MarvinUserWarning(UserWarning, MarvinWarning):
    """The primary warning class."""
    pass


def _doLocal(instance, **kwargs):
    """Does local things."""

    filename = kwargs.get('filename', None)
    plateifu = kwargs.get('plateifu', None)

    if filename:

        if os.path.exists(filename):
            kwargs['mode'] = 'local'
            return kwargs
        else:
            raise MarvinError('input file {0} not found'.format(filename))

    elif plateifu:

        dbStatus = testDbConnection(config.session)

        if dbStatus['good']:
            kwargs['mode'] = 'local'
            return kwargs
        else:
            warnings.warn(
                'DB connection failed with error: {0}.'
                .format(dbStatus['error']), MarvinUserWarning)

            fullpath = instance._getFullPath(**kwargs)

            if os.path.exists(fullpath):
                kwargs['mode'] = 'local'
                kwargs['filename'] = fullpath
                return kwargs
            else:
                if config.download:
                    raise NotImplementedError('sdsssync not yet implemented')
                    # When implemented, this should download the data and
                    # then kwargs['filename'] = downloaded_path and
                    # kwargs['mode'] = local
                else:
                    raise MarvinError('this is the end of the road. Try '
                                      'using some reasonable inputs.')


def _doRemote(**kwargs):
    """Do remote things."""

    filename = kwargs.get('filename', None)

    if filename:
        raise MarvinError('filename not allowed in remote mode.')
    else:
        kwargs['mode'] = 'remote'
        return kwargs


class MarvinToolsClass(object):

    def __new__(cls, *args, **kwargs):

        me = object.__new__(cls, *args, **kwargs)

        me._kwargs = kwargs

        filename = kwargs.get('filename', None)
        mangaid = kwargs.get('mangaid', None)
        plateifu = kwargs.get('plateifu', None)
        mode = kwargs.get('mode', None)

        if mode is None:
            mode = config.mode
        me._kwargs['mode'] = mode

        args = [filename, plateifu, mangaid]
        errmsg = 'Enter filename, plateifu, or mangaid!'
        assert any(args), errmsg
        assert sum([bool(arg) for arg in args]) == 1, errmsg

        if mangaid:
            if mangaid in mangaid_to_plateifu:
                me._kwargs['plateifu'] = mangaid_to_plateifu[mangaid]
                me._kwargs['mangaid'] = None
            else:
                raise MarvinError('mangaid={0} not found in the dictionary'
                                  .format(mangaid))

        if mode == 'local':
            me._kwargs = _doLocal(me, **kwargs)
        elif mode == 'remote':
            me._kwargs = _doRemote(**kwargs)
        elif mode == 'auto':
            try:
                me._kwargs = _doLocal(me, **kwargs)
            except:
                warnings.warn('local mode failed. Trying remote now.',
                              MarvinUserWarning)
                me._kwargs = _doRemote(**kwargs)

        return me

    def _getFullPath(self, **kwargs):

        if not Path:
            raise MarvinError('sdss_access is not installed')
        else:
            self._Path = Path

        # - check only one mangaid, filename, or plateifu
        # - if mangaid
        # -     if key exists in mangaid_to_plateifu
        # -         plateifu=mangaid_to_plateifu[mangaid]
        # -     else
        # -         FAILS
        #
        # if local
        #     if filename
        #         if exists
        #             mode=local+filename
        #         else
        #             FAILS
        #     else (plateifu)
        #         if DB:
        #             mode=local+plateifu
        #         else
        #             if file exists in tree
        #                 mode=local+full file path
        #             else:
        #                 if download_all
        #                     sdsssync
        #                     mode=local+full file path
        #                 else
        #                     FAILS
        # elif remote
        #     if filename
        #         FAILS
        #     else (plateifu)
        #         mode=remote+plate_ifu
        # elif auto
        #     try:
        #         local
        #     except:
        #         remote

        # if mode == 'local':
        #     if filename:
        #         kwargs['mode'] = 'local'
        #         if os.path.exists(filename):
        #             pass
        #         else:
        #             if not mangaid and not plateifu:
        #                 raise MarvinError('you did not provide any input!')
        #             elif mangaid and plateifu:
        #                 raise MarvinError('provide mangaid or plateifu')
        #             elif mangaid:
        #                 try:
        #                     plateifu = mangaid_to_plateifu[mangaid]
        #                     kwargs['mangaid'] = None
        #                     kwargs['plateifu'] = plateifu
        #                 except KeyError:
        #                     raise MarvinError(
        #                         'mangaid={0} not found in the dictionary'
        #                         .format(mangaid))
        #
        #         return me(*args, **kwargs)
