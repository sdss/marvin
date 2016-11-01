#!/usr/bin/env python
# encoding: utf-8


# Created by Brian Cherinka on 2016-05-10 20:17:52
# Licensed under a 3-clause BSD license.

# Revision History:
#     Initial Version: 2016-05-10 20:17:52 by Brian Cherinka
#     Last Modified On: 2016-05-10 20:17:52 by Brian


from __future__ import print_function, division
from marvin.core.core import MarvinToolsClass
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin import config
import numpy as np
import warnings
from functools import wraps

try:
    from sdss_access import RsyncAccess, AccessError
    from sdss_access.path import Path
except ImportError:
    Path = None
    RsyncAccess = None

__all__ = ['Image']


class Image(MarvinToolsClass):
    '''A class to interface with MaNGA images.

     - DRP optical images
     - DAP MAP images
     - NSA preimaging
     mangafile = 'optical' or 'map' or 'nsa'

    - maybe Image is new Core base class instead of MarvinToolsClass?
    - and we have new classes for MapImage, DRPImage, NSAImage?

    '''

    def __init__(self, *args, **kwargs):

        self.data_origin = None
        self._drpver = kwargs.get('drpver', config.drpver)
        self._dapver = kwargs.get('dapver', config.dapver)
        self._mplver = kwargs.get('mplver', config.mplver)
        self.download = kwargs.get('download', config.download)

        MarvinToolsClass.__init__(self, *args, **kwargs)

        if self.mode == 'local':
            if self.filename:
                # do any file stuff
                pass
            else:
                # do any db stuff
                self.plate, self.ifu = self.plateifu.split('-')

        else:
            # do any remote stuff
            pass

    def __repr__(self):
        '''Image representation.'''
        return '<Marvin Image (plate={0}, ifu={1}, drpver={2})>'.format(self.plate, self.ifu, self.drpver)

    def getByPlate(self):
        pass

    def getByList(self):
        pass

    def getRandom(self):
        pass


