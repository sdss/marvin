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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

from astropy.io import fits
from marvin.tools.mixins import MMAMixIn
from marvin.core.exceptions import MarvinError, MarvinWarning
from marvin.utils.general import getWCSFromPng, Bundle, Cutout
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

    This class represents a MaNGA image object initialised either
    from a file, or remotely via the Marvin API.

    TODO: what kinds of images should this handle? optical, maps, nsa preimaging?
    TODO: should this be subclasses into different kinds of images?  DRPImage, MapImage, NSAImage?

    Attributes:
        header (`astropy.io.fits.Header`):
            The header of the datacube.
        ra,dec (float):
            Coordinates of the target.
        wcs (`astropy.wcs.WCS`):
            The WCS solution for this plate
        bundle (object):
            A Bundle of fibers associated with the IFU
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

        # initialize attributes
        self._init_attributes()

        # create the hex bundle
        self.bundle = Bundle(self.ra, self.dec, plateifu=self.plateifu, size=int(str(self.ifu)[:-2]))

    def __repr__(self):
        '''Image representation.'''
        return '<Marvin Image (plateifu={0}, mode={1}, data-origin={2})>'.format(repr(self.plateifu), repr(self.mode), repr(self.data_origin))

    def _init_attributes(self):
        ''' Initialize some attributes '''
        self.header = fits.header.Header(self.data.info)
        self.ra = float(self.header["RA"])
        self.dec = float(self.header["DEC"])
        self.wcs = getWCSFromPng(image=self.data)

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

    def plot(self, return_figure=True, dpi=100, fibers=None, skies=None, **kwargs):
        ''' Creates a Matplotlib plot the image

        Parameters:
            fibers (bool):
                If True, overlays the fiber positions. Default is False.
            skies (bool):
                If True, overlays the sky fibers if possible. Default is False.
            return_figure (bool):
                If True, returns the figure axis object.  Default is True
            dpi (int):
                The dots per inch for the matplotlib figure
            kwargs:
                Keyword arguments for overlay_fibers and overlay_skies
        '''
        pix_size = np.array(self.data.size)
        fig = plt.figure(figsize=pix_size / dpi, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        plt.axis('off')
        fig.add_axes(ax)

        ax.imshow(self.data, origin='upper', aspect='auto')
        ax.autoscale(False)

        # overlay the IFU fibers
        if fibers:
            self.overlay_fibers(ax, return_figure=return_figure, skies=skies, **kwargs)

        if return_figure:
            return ax

    def overlay_fibers(self, ax, diameter=None, skies=None, return_figure=True, **kwargs):
        """ Overlay the individual fibers within an IFU on a plot.

        Parameters:
            ax (Axis):
                The matplotlib axis object
            diameter (float):
                The fiber diameter in arcsec. Default is 2".
            skies (bool):
                Set to True to additionally overlay the sky fibers. Default if False
            return_figure (bool):
                If True, returns the figure axis object.  Default is True
            kwargs:
                Any keyword arguments accepted by Matplotlib EllipseCollection
        """

        # check the diameter
        if diameter:
            assert isinstance(diameter, (float, int)), 'diameter must be a number'
        diameter = (diameter or 2.0) / float(self.header['SCALE'])

        # get the fiber pixel coordinates
        fibers = self.bundle.fibers[:, [1, 2]]
        fiber_pix = self.wcs.wcs_world2pix(fibers, 1)

        # some matplotlib kwargs
        kwargs['edgecolor'] = kwargs.get('edgecolor', 'Orange')
        kwargs['facecolor'] = kwargs.get('facecolor', 'none')
        kwargs['linewidth'] = kwargs.get('linewidth', 0.4)

        ec = EllipseCollection(diameter, diameter, 0.0, units='xy',
                               offsets=fiber_pix, transOffset=ax.transData,
                               **kwargs)
        ax.add_collection(ec)

        # overlay the sky fibers
        if skies:
            self.overlay_skies(ax, diameter=diameter, return_figure=return_figure, **kwargs)

        if return_figure:
            return ax

    def overlay_hexagon(self, ax, return_figure=True, **kwargs):
        """ Overlay the IFU hexagon on a plot

        Parameters:
            ax (Axis):
                The matplotlib axis object
            return_figure (bool):
                If True, returns the figure axis object.  Default is True
            kwargs:
                Any keyword arguments accepted by Matplotlib plot
        """

        # get IFU hexagon pixel coordinates
        hexagon_pix = self.wcs.wcs_world2pix(self.bundle.hexagon, 1)
        # reconnect the last point to the first point.
        hexagon_pix = np.concatenate((hexagon_pix, [hexagon_pix[0]]), axis=0)

        # some matplotlib kwargs
        kwargs['color'] = kwargs.get('color', 'magenta')
        kwargs['linestyle'] = kwargs.get('linestyle', 'solid')
        kwargs['linewidth'] = kwargs.get('linewidth', 0.8)

        ax.plot(hexagon_pix[:, 0], hexagon_pix[:, 1], **kwargs)

        if return_figure:
            return ax

    def overlay_skies(self, ax, diameter=None, return_figure=True, **kwargs):
        """ Overlay the sky fibers on a plot

        Parameters:
            ax (Axis):
                The matplotlib axis object
            diameter (float):
                The fiber diameter in arcsec
            return_figure (bool):
                If True, returns the figure axis object.  Default is True
            kwargs:
                Any keyword arguments accepted by Matplotlib EllipseCollection
        """

        # check for sky coordinates
        if self.bundle.skies is None:
            self.bundle.get_sky_coordinates()

        # check the diameter
        if diameter:
            assert isinstance(diameter, (float, int)), 'diameter must be a number'
        diameter = (diameter or 2.0) / float(self.header['SCALE'])

        # get sky fiber pixel positions
        fiber_pix = self.wcs.wcs_world2pix(self.bundle.skies, 1)
        outside_range = ((fiber_pix < 0) | (fiber_pix > self.data.size[0])).any()
        if outside_range:
            raise MarvinError('Cannot overlay sky fibers.  Image is too small.  '
                              'Please retrieve a bigger image cutout')

        # some matplotlib kwargs
        kwargs['edgecolor'] = kwargs.get('edgecolor', 'Orange')
        kwargs['facecolor'] = kwargs.get('facecolor', 'none')
        kwargs['linewidth'] = kwargs.get('linewidth', 0.7)

        # draw the sky fibers
        ec = EllipseCollection(diameter, diameter, 0.0, units='xy',
                               offsets=fiber_pix, transOffset=ax.transData,
                               **kwargs)
        ax.add_collection(ec)

        # Add a larger circle to help identify the sky fiber locations in large images.
        if (self.data.size[0] > 1000) or (self.data.size[1] > 1000):
            ec = EllipseCollection(diameter * 5, diameter * 5, 0.0, units='xy',
                                   offsets=fiber_pix, transOffset=ax.transData,
                                   **kwargs)
            ax.add_collection(ec)

        if return_figure:
            return ax

    def get_new_cutout(self, width, height, scale=None, **kwargs):
        ''' Get a new Image Cutout using Skyserver '''

        cutout = Cutout(self.ra, self.dec, width, height, scale=scale, **kwargs)
        self.data = cutout.image
        self._init_attributes()

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
