# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-06-21 17:01:09
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-06-22 12:08:48

from __future__ import absolute_import, division, print_function

import glob
import os
import time
import warnings

from marvin.core.exceptions import MarvinError, MarvinUserWarning

import sdss_access.path
import sdss_access.sync


class VACMixIn(object):
    """This mixin allows for integrating a VAC into Marvin."""

    # The name of the VAC.
    name = None

    def __init__(self):

        if 'MANGA_SANDBOX' not in os.environ:
            os.environ['MANGA_SANDBOX'] = os.path.join(os.getenv('SAS_BASE_DIR'),
                                                       'mangawork/manga/sandbox')

        if not sdss_access.sync.RsyncAccess:
            raise MarvinError('sdss_access is not installed')
        else:
            self.rsync_access = sdss_access.sync.RsyncAccess()

    def get_vacs(self, parent_object):

        class VACContainer(object):

            def __getitem__(self, value):
                return getattr(self, value)

            def __iter__(self):
                for value in self.__dir__():
                    if not value.startswith('_'):
                        yield value

        vac_container = VACContainer()

        for subvac in VACMixIn.__subclasses__():
            # We need to set sv=subvac in the lambda function to prevent
            # a cell-var-from-loop issue.
            setattr(VACContainer, subvac.name,
                    property(lambda self, sv=subvac: sv().get_data(parent_object)))

        return vac_container

    def download_vac(self, path_params={}, verbose=True):
        """Download the VAC using rsync and returns the local path."""

        assert self.name in self.rsync_access.templates, 'VAC path has not been set in the tree.'

        if verbose:
            warnings.warn('downloading file for VAC {0!r}'.format(self.name), MarvinUserWarning)

        self.rsync_access.remote()
        self.rsync_access.add(self.name, **path_params)
        self.rsync_access.set_stream()
        self.rsync_access.commit()
        paths = self.rsync_access.get_paths()

        # adding a millisecond pause for download to finish and file existence to register
        time.sleep(0.001)

        return paths[0]  # doing this for single files, may need to change

    def set_sandbox_path(self, relative_path):
        """Downloads a file using its path relative to ``MANGA_SANDBOX``."""

        self.rsync_access.templates[self.name] = os.path.join('$MANGA_SANDBOX', relative_path)

    def get_path(self, path_params={}):
        """Returns the local path of the VAC file."""

        # The full path could be a pattern (e.g., with stars), so we use glob
        # to get a unique path
        pattern = self.rsync_access.full(self.name, **path_params)

        files = glob.glob(pattern)
        if len(files) > 0:
            return files[0]
        else:
            return pattern

    def file_exists(self, path_params={}):
        """Check whether a file exists."""

        if os.path.exists(self.get_path(path_params=path_params)):
            return True

        return False
