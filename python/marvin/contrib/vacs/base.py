# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-06-21 17:01:09
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-06-22 11:22:39

from __future__ import print_function, division, absolute_import
import abc
import six
import time
import os
from marvin.core.exceptions import MarvinError
from sdss_access.sync import RsyncAccess


class VACMixIn(object, six.with_metaclass(abc.ABCMeta)):
    ''' This mixin allows for integrating a VAC into Marvin '''

    @abc.abstractmethod
    def _get_from_file(self):
        ''' This method controls accessing a VAC from a local file '''
        pass

    def download_vac(self, name=None, **path_params):
        """Download the VAC using rsync """

        if 'MANGA_SANDBOX' not in os.environ:
            os.environ['MANGA_SANDBOX'] = os.path.join(os.getenv("SAS_BASE_DIR"), 'mangawork/manga/sandbox')

        if not RsyncAccess:
            raise MarvinError('sdss_access is not installed')
        else:
            rsync_access = RsyncAccess()
            rsync_access.remote()
            rsync_access.add(name, **path_params)
            rsync_access.set_stream()
            rsync_access.commit()
            paths = rsync_access.get_paths()
            time.sleep(0.001)  # adding a millisecond pause for download to finish and file extistence to register
            self.filename = paths[0]  # doing this for single files, may need to change
