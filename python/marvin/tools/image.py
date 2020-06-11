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
import random
import itertools
import requests
import PIL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

from astropy.io import fits
import marvin
from marvin.tools.mixins import MMAMixIn
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.utils.general import (getWCSFromPng, Bundle, Cutout, target_is_mastar,
                                  get_plates, check_versions)
try:
    from sdss_access import HttpAccess
except ImportError:
    HttpAccess = None

if sys.version_info.major == 2:
    from cStringIO import StringIO as stringio
else:
    from io import BytesIO as stringio

__all__ = ['Image']


class Image(MMAMixIn):
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
        if self.ra and self.dec:
            self.bundle = Bundle(self.ra, self.dec, plateifu=self.plateifu, size=int(str(self.ifu)[:-2]))

    def __repr__(self):
        '''Image representation.'''
        return '<Marvin Image (plateifu={0}, mode={1}, data-origin={2})>'.format(repr(self.plateifu), repr(self.mode), repr(self.data_origin))

    def _init_attributes(self):
        ''' Initialize some attributes '''

        self.ra = None
        self.dec = None

        # try to create a header
        try:
            self.header = fits.header.Header(self.data.info)
        except Exception:
            warnings.warn('No proper header found image', MarvinUserWarning)
            self.header = None

        # try to set the RA, Dec
        if self.header:
            self.ra = float(self.header["RA"]) if 'RA' in self.header else None
            self.dec = float(self.header["DEC"]) if 'DEC' in self.header else None

        # try to set the WCS
        try:
            self.wcs = getWCSFromPng(image=self.data)
        except MarvinError:
            self.wcs = None
            warnings.warn('No proper WCS info for this image')

    def _get_image_dir(self):
        ''' Gets the correct images directory by release and mastar target '''

        # all images in MPL-4 are in stack dirs
        if self.release == 'MPL-4':
            return 'stack'

        # get the appropriate image directory
        is_mastar = target_is_mastar(self.plateifu, drpver=self._drpver)
        image_dir = 'mastar' if is_mastar else 'stack'
        return image_dir

    def _getFullPath(self):
        """Returns the full path of the file in the tree."""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')
        dir3d = self._get_image_dir()

        # # use version to toggle old/new images path
        # isMPL8 = check_versions(self._drpver, 'v2_5_3')
        # name = 'mangaimagenew' if isMPL8 else 'mangaimage'
        name = 'mangaimage'

        return super(Image, self)._getFullPath(name, ifu=ifu, dir3d=dir3d,
                                               drpver=self._drpver, plate=plate)

    def download(self):
        """Downloads the image using sdss_access - Rsync,"""

        if not self.plateifu:
            return None

        plate, ifu = self.plateifu.split('-')
        dir3d = self._get_image_dir()

        # # use version to toggle old/new images path
        # isMPL8 = check_versions(self._drpver, 'v2_5_3')
        # name = 'mangaimagenew' if isMPL8 else 'mangaimage'
        name = 'mangaimage'

        return super(Image, self).download(name, ifu=ifu, dir3d=dir3d,
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
        response = requests.get(self.url)
        if not response.ok:
            raise MarvinError('Error: remote filepath {0} does not exist'.format(filepath))
        else:
            fileobj = stringio(response.content)
            self.data = self._open_image(fileobj, filepath=self.url)

    @property
    def url(self):
        if not HttpAccess:
            raise MarvinError('sdss_access not installed')

        filepath = self._getFullPath()
        http = HttpAccess(verbose=False)
        url = http.url("", full=filepath)
        return url

    @staticmethod
    def _open_image(fileobj, filepath=None):
        ''' Open the Image using PIL '''

        try:
            image = PIL.Image.open(fileobj)
        except IOError as e:
            warnings.warn('Error: cannot open image', MarvinUserWarning)
            image = None
        else:
            image.filename = filepath or fileobj

        return image

    def show(self):
        ''' Show the image '''
        if self.data:
            self.data.show()

    def save(self, filename, filetype='png', **kwargs):
        ''' Save the image to a file

        This only saves the original image.  To save the Matplotlib plot, use
        the savefig method on the matplotlib.pyplot.figure object

        Parameters:
            filename (str):
                The filename of the output image
            filetype (str):
                The filetype, e.g. png
            kwargs:
                Additional keyword arguments to the PIL.Image.save method

        '''
        __, fileext = os.path.splitext(filename)
        assert filetype or fileext, 'Filename must have an extension or specify the filetype'

        if self.data:
            self.data.save(filename, format=filetype, **kwargs)

    def plot(self, return_figure=True, dpi=100, with_axes=None, fibers=None, skies=None, **kwargs):
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
            with_axes (bool):
                If True, plots the image with axes
            kwargs:
                Keyword arguments for overlay_fibers and overlay_skies
        '''

        pix_size = np.array(self.data.size)
        figsize = np.ceil(pix_size / dpi)
        if with_axes:
            fig = plt.figure()
            ax = plt.subplot(projection=self.wcs)
            ax.set_xlabel('Declination')
            ax.set_ylabel('Right Ascension')
            aspect = None
        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0., 0., 1., 1.], projection=self.wcs)
            ax.set_axis_off()
            plt.axis('off')
            aspect = 'auto'

        ax.imshow(self.data, origin='lower', aspect=aspect)

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

        if self.wcs is None:
            raise MarvinError('No WCS found.  Cannot overlay fibers.')

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

        if self.wcs is None:
            raise MarvinError('No WCS found.  Cannot overlay hexagon.')

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

        if self.wcs is None:
            raise MarvinError('No WCS found.  Cannot overlay sky fibers.')

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
        ''' Get a new Image Cutout using Skyserver

        Replaces the current Image with a new image cutout.  The
        data, header, and wcs attributes are updated accordingly.

        Parameters:
            width (int):
                Cutout image width in arcsec
            height (int):
                Cutout image height in arcsec
            scale (float):
                arcsec/pixel scale of the image
            kwargs:
                Any additional keywords for Cutout

        '''

        cutout = Cutout(self.ra, self.dec, width, height, scale=scale, **kwargs)
        self.data = cutout.image
        self._init_attributes()

    @classmethod
    def from_list(cls, values, release=None):
        ''' Generate a list of Marvin Image objects

        Class method to generate a list of Marvin Images from an
        input list of targets

        Parameters:
            values (list):
                A list of target ids (i.e. plateifus, mangaids, or filenames)
            release (str):
                The release of Images to get

        Returns:
            a list of Marvin Image objects

        Example:
            >>> from marvin.tools.image import Image
            >>> targets = ['8485-1901', '7443-1201']
            >>> images = Image.from_list(targets)

        '''
        images = []
        for item in values:
            images.append(cls(item, release=release))
        return images

    @classmethod
    def by_plate(cls, plateid, minis=None, release=None):
        ''' Generate a list of Marvin Images by plate

        Class method to generate a list of Marvin Images from
        a single plateid.

        Parameters:
            plateid (int):
                The plate id to grab
            minis (bool):
                If True, includes the mini-bundles
            release (str):
                The release of Images to get

        Returns:
            a list of Marvin Image objects

        Example:
            >>> from marvin.tools.image import Image
            >>> images = Image.by_plate(8485)
        '''
        ifus = cls._get_ifus(minis=minis)
        plateifus = ['{0}-{1}'.format(plateid, i) for i in ifus]
        images = cls.from_list(plateifus, release=release)
        return images

    @classmethod
    def get_random(cls, num=5, minis=None, release=None):
        ''' Generate a set of random Marvin Images

        Class method to generate a random list of Marvin Images

        Parameters:
            num (int):
                The number to grab. Default is 5
            minis (bool):
                If True, includes the mini-bundles
            release (str):
                The release of Images to get

        Returns:
            a list of Marvin Image objects

        Example:
            >>> from marvin.tools.image import Image
            >>> images = Image.get_random(5)
        '''
        ifus = cls._get_ifus(minis=minis)
        plates = get_plates(release=release)
        rand_samp = random.sample(list(itertools.product(map(str, plates), ifus)), num)
        plateifus = ['-'.join(r) for r in rand_samp]
        images = cls.from_list(plateifus, release=release)
        return images

    def getCube(self):
        """Returns the :class:`~marvin.tools.cube.Cube` for this Image."""

        return marvin.tools.cube.Cube(plateifu=self.plateifu,
                                      release=self.release)

    def getMaps(self, **kwargs):
        """Retrieves the DAP :class:`~marvin.tools.maps.Maps` for this Image.

        If called without additional ``kwargs``, :func:`getMaps` will initialize
        the :class:`~marvin.tools.maps.Maps` using the ``plateifu`` of this
        :class:`~marvin.tools.image.Image`. Otherwise, the ``kwargs`` will be
        passed when initialising the :class:`~marvin.tools.maps.Maps`.

        """

        if len(kwargs.keys()) == 0 or 'filename' not in kwargs:
            kwargs.update({'plateifu': self.plateifu, 'release': self._release})

        maps = marvin.tools.maps.Maps(**kwargs)
        return maps

