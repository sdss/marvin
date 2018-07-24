#!/usr/bin/env python3
# encoding: utf-8
#
# pickle.py
#
# Created by José Sánchez-Gallego on 7 Oct 2016.


from __future__ import absolute_import, division, print_function

import os
import warnings

from six import string_types

from marvin.core.exceptions import MarvinError, MarvinUserWarning


try:
    import cPickle as pickle
except:
    import pickle





def save(obj, path=None, overwrite=False):
    """Pickles the object.

    If ``path=None``, uses the default location of the file in the tree
    but changes the extension of the file to ``.mpf`` (MaNGA Pickle File).
    Returns the path of the saved pickle file.

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

    from ..tools.core import MarvinToolsClass

    if path is None:
        assert isinstance(obj, MarvinToolsClass), 'path=None is only allowed for core objects.'
        path = obj._getFullPath()
        assert isinstance(path, string_types), 'path must be a string.'
        if path is None:
            raise MarvinError('cannot determine the default path in the '
                              'tree for this file. You can overcome this '
                              'by calling save with a path keyword with '
                              'the path to which the file should be saved.')

        # Replaces the extension (normally fits) with mpf
        if '.fits.gz' in path:
            path = path.strip('.fits.gz')
        else:
            path = os.path.splitext(path)[0]

        path += '.mpf'

    path = os.path.realpath(os.path.expanduser(path))

    if os.path.isdir(path):
        raise MarvinError('path must be a full route, including the filename.')

    if os.path.exists(path) and not overwrite:
        warnings.warn('file already exists. Not overwriting.', MarvinUserWarning)
        return

    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    try:
        with open(path, 'wb') as fout:
            pickle.dump(obj, fout, protocol=-1)
    except Exception as ee:
        if os.path.exists(path):
            os.remove(path)
        raise MarvinError('error found while pickling: {0}'.format(str(ee)))

    return path


def restore(path, delete=False):
    """Restores a MarvinToolsClass object from a pickled file.

    If ``delete=True``, the pickled file will be removed after it has been
    unplickled. Note that, for objects with ``data_origin='file'``, the
    original file must exists and be in the same path as when the object
    was first created.

    """

    assert isinstance(path, string_types), 'path must be a string.'

    path = os.path.realpath(os.path.expanduser(path))

    if os.path.isdir(path):
        raise MarvinError('path must be a full route, including the filename.')

    if not os.path.exists(path):
        raise MarvinError('the path does not exists.')

    try:
        with open(path, 'rb') as fin:
            obj = pickle.load(fin)
    except Exception as ee:
        raise MarvinError('something went wrong unplicking the object: {0}'
                          .format(str(ee)))

    if delete is True:
        os.remove(path)

    return obj
