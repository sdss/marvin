#!/usr/bin/env python
# encoding: utf-8


# Created by Brian Cherinka on 2016-05-10 20:17:52
# Licensed under a 3-clause BSD license.

# Revision History:
#     Initial Version: 2016-05-10 20:17:52 by Brian Cherinka
#     Last Modified On: 2016-05-10 20:17:52 by Brian


from __future__ import absolute_import, division, print_function

import os
import sys
import warnings
import requests
import PIL

from astropy.io import fits
from marvin.tools.mixins import MMAMixIn
from marvin.core.exceptions import MarvinError, MarvinWarning
from marvin.utils.general import getWCSFromPng, Bundle
try:
    from sdss_access import HttpAccess
except ImportError:
    HttpAccess = None

if sys.version_info.major == 2:
    from cStringIO import StringIO as stringio
else:
    from io import BytesIO as stringio

__all__ = ['Image']


class Image(MMAMixIn, object):
    '''A class to interface with MaNGA images.

     - DRP optical images
     - DAP MAP images
     - NSA preimaging
     mangafile = 'optical' or 'map' or 'nsa'

    - maybe Image is new Core base class instead of MarvinToolsClass?
    - and we have new classes for MapImage, DRPImage, NSAImage?

    '''

    def __init__(self, input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None, download=None):

        MMAMixIn.__init__(self, input=input, filename=filename, mangaid=mangaid,
                          plateifu=plateifu, mode=mode, data=data, release=release,
                          download=download, ignore_db=True)

        if self.data_origin == 'file':
            self._load_image_from_file()
        elif self.data_origin == 'db':
            raise MarvinError('Images cannot currently be accessed from the db')
        elif self.data_origin == 'api':
            self._load_image_from_api()

        # initialize
        self.header = fits.header.Header(self.data.info)
        self.ra = self.header["RA"]
        self.dec = self.header["DEC"]
        self.wcs = getWCSFromPng(image=self.data)

        # create the hex bundle
        self.bundle = Bundle(self.ra, self.dec, int(str(self.ifu)[:-2]))

    def __repr__(self):
        '''Image representation.'''
        return '<Marvin Image (plateifu={0}, mode={1}, data-origin={2})>'.format(repr(self.plateifu), repr(self.mode), repr(self.data_origin))

    def _getFullPath(self):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return super(Image, self)._getFullPath('mangaimage', ifu=ifu, dir3d='stack',
                                               drpver=self._drpver, plate=plate)

    def download(self):
        """Downloads the cube using sdss_access - Rsync,"""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')

        return super(Image, self).download('mangaimage', ifu=ifu, dir3d='stack',
                                           drpver=self._drpver, plate=plate)

    def _load_image_from_file(self):
        ''' Load an image from a local file '''

        filepath = self._getFullPath()
        if os.path.exists(filepath):
            self._filepath = filepath
            self.data = self._open_image(filepath)
        else:
            raise MarvinError('Error: local filepath {0} does not exist. '.format(filepath))

    def _load_image_from_api(self):
        ''' Load an image from a remote location '''

        filepath = self._getFullPath()
        if not HttpAccess:
            raise MarvinError('Cannot get ')

        http = HttpAccess(verbose=False)
        url = http.url("", full=filepath)
        response = requests.get(url)
        if not response.ok:
            raise MarvinError('Error: remote filepath {0} does not exist'.format(filepath))
        else:
            fileobj = stringio(response.content)
            self.data = self._open_image(fileobj, filepath=url)

    @staticmethod
    def _open_image(fileobj, filepath=None):
        ''' Open the Image using PIL '''

        try:
            image = PIL.Image.open(fileobj)
        except IOError as e:
            warnings.warn('Error: cannot open image', MarvinWarning)
            image = None
        else:
            image.filename = filepath or fileobj

        return image

    def show(self):
        ''' Show the image '''
        if self.data:
            self.data.show()

    def save(self, filename, filetype=None, **kwargs):
        ''' Save the image '''
        if self.data:
            self.data.save(filename, format=filetype, **kwargs)

    def get_by_plate(self):
        pass

    @classmethod
    def from_list(cls, values):
        images = []
        for item in values:
            images.append(cls(item))
        return images

    def get_random(self):
        pass
