#!/usr/bin/env python
# encoding: utf-8
#
# Licensed under a 3-clause BSD license.
#
#
# map.py
#
# Created by José Sánchez-Gallego on 26 Jun 2016.
#
# Includes code from mangadap.plot.maps.py licensed under the following 3-clause BSD license.
#
# Copyright (c) 2015, SDSS-IV/MaNGA Pipeline Group
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other materials provided with
# the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to
# endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from distutils import version
import os
import sys
import warnings

from astropy.io import fits
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import marvin
import marvin.api.api
import marvin.core.marvin_pickle
import marvin.core.exceptions
import marvin.tools.maps
import marvin.utils.plot.colorbar as colorbar

if 'seaborn' in sys.modules:
    import seaborn as sns
else:
    plt.style.use('seaborn-darkgrid')

try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None


class Map(object):
    """Describes a single DAP map in a Maps object.

    Unlike a ``Maps`` object, which contains all the information from a DAP
    maps file, this class represents only one of the multiple 2D maps contained
    within. For instance, ``Maps`` may contain emission line maps for multiple
    channels. A ``Map`` would be, for example, the map for ``emline_gflux`` and
    channel ``ha_6564``.

    A ``Map`` is basically a set of three Numpy 2D arrays (``value``, ``ivar``,
    and ``mask``), with extra information and additional methods for
    functionality.

    ``Map`` objects are not intended to be initialised directly, at least
    for now. To get a ``Map`` instance, use the
    :func:`~marvin.tools.maps.Maps.getMap` method.

    Parameters:
        maps (:class:`~marvin.tools.maps.Maps` object):
            The :class:`~marvin.tools.maps.Maps` instance from which we
            are extracting the ``Map``.
        property_name (str):
            The category of the map to be extractred. E.g., `'emline_gflux'`.
        channel (str or None):
            If the ``property`` contains multiple channels, the channel to use,
            e.g., ``ha_6564'. Otherwise, ``None``.

    """

    def __init__(self, maps, property_name, channel=None):

        assert isinstance(maps, marvin.tools.maps.Maps)

        self.maps = maps
        self.property_name = property_name.lower()
        self.channel = channel.lower() if channel else None
        self.shape = self.maps.shape

        self.release = maps.release

        self.maps_property = self.maps.properties[self.property_name]
        if (self.maps_property is None or
                (self.maps_property.channels is not None and
                 self.channel not in self.maps_property.channels)):
            raise marvin.core.exceptions.MarvinError(
                'invalid combination of property name and channel.')

        self.value = None
        self.ivar = None
        self.mask = None

        self.header = None
        self.unit = None

        if maps.data_origin == 'file':
            self._load_map_from_file()
        elif maps.data_origin == 'db':
            self._load_map_from_db()
        elif maps.data_origin == 'api':
            self._load_map_from_api()

        self.masked = np.ma.array(self.value, mask=(self.mask > 0
                                                    if self.mask is not None else False))

    def __repr__(self):

        return ('<Marvin Map (plateifu={0.maps.plateifu!r}, property={0.property_name!r}, '
                'channel={0.channel!r})>'.format(self))

    @property
    def snr(self):
        """Returns the signal-to-noise ratio for each spaxel in the map."""

        return np.abs(self.value * np.sqrt(self.ivar))

    def _load_map_from_file(self):
        """Initialises the Map from a ``Maps`` with ``data_origin='file'``."""

        self.header = self.maps.data[self.property_name].header

        if self.channel is not None:
            channel_idx = self.maps_property.channels.index(self.channel)
            self.value = self.maps.data[self.property_name].data[channel_idx]
            if self.maps_property.ivar:
                self.ivar = self.maps.data[self.property_name + '_ivar'].data[channel_idx]
            if self.maps_property.mask:
                self.mask = self.maps.data[self.property_name + '_mask'].data[channel_idx]
        else:
            self.value = self.maps.data[self.property_name].data
            if self.maps_property.ivar:
                self.ivar = self.maps.data[self.property_name + '_ivar'].data
            if self.maps_property.mask:
                self.mask = self.maps.data[self.property_name + '_mask'].data

        if isinstance(self.maps_property.unit, list):
            self.unit = self.maps_property.unit[channel_idx]
        else:
            self.unit = self.maps_property.unit

        return

    def _load_map_from_db(self):
        """Initialises the Map from a ``Maps`` with ``data_origin='db'``."""

        mdb = marvin.marvindb

        if not mdb.isdbconnected:
            raise marvin.core.exceptions.MarvinError('No db connected')

        if sqlalchemy is None:
            raise marvin.core.exceptions.MarvinError('sqlalchemy required to access the local DB.')

        if version.StrictVersion(self.maps._dapver) <= version.StrictVersion('1.1.1'):
            table = mdb.dapdb.SpaxelProp
        else:
            table = mdb.dapdb.SpaxelProp5

        fullname_value = self.maps_property.fullname(channel=self.channel)
        value = mdb.session.query(getattr(table, fullname_value)).filter(
            table.file_pk == self.maps.data.pk).order_by(table.spaxel_index).all()
        self.value = np.array(value).reshape(self.shape).T

        if self.maps_property.ivar:
            fullname_ivar = self.maps_property.fullname(channel=self.channel, ext='ivar')
            ivar = mdb.session.query(getattr(table, fullname_ivar)).filter(
                table.file_pk == self.maps.data.pk).order_by(table.spaxel_index).all()
            self.ivar = np.array(ivar).reshape(self.shape).T

        if self.maps_property.mask:
            fullname_mask = self.maps_property.fullname(channel=self.channel, ext='mask')
            mask = mdb.session.query(getattr(table, fullname_mask)).filter(
                table.file_pk == self.maps.data.pk).order_by(table.spaxel_index).all()
            self.mask = np.array(mask).reshape(self.shape).T

        # Gets the header
        hdus = self.maps.data.hdus
        header_dict = None
        for hdu in hdus:
            if self.maps_property.name.upper() == hdu.extname.name.upper():
                header_dict = hdu.header_to_dict()
                break

        if not header_dict:
            warnings.warn('cannot find the header for property {0}.'
                          .format(self.maps_property.name),
                          marvin.core.exceptions.MarvinUserWarning)
        else:
            self.header = fits.Header(header_dict)

        self.unit = self.maps_property.unit

    def _load_map_from_api(self):
        """Initialises the Map from a ``Maps`` with ``data_origin='api'``."""

        url = marvin.config.urlmap['api']['getmap']['url']

        url_full = url.format(
            **{'name': self.maps.plateifu,
               'property_name': self.property_name,
               'channel': self.channel,
               'bintype': self.maps.bintype,
               'template_kin': self.maps.template_kin})

        try:
            response = marvin.api.api.Interaction(url_full,
                                                  params={'release': self.maps._release})
        except Exception as ee:
            raise marvin.core.exceptions.MarvinError(
                'found a problem when getting the map: {0}'.format(str(ee)))

        data = response.getData()

        if data is None:
            raise marvin.core.exceptions.MarvinError(
                'something went wrong. Error is: {0}'.format(response.results['error']))

        self.value = np.array(data['value'])
        self.ivar = np.array(data['ivar']) if data['ivar'] is not None else None
        self.mask = np.array(data['mask']) if data['mask'] is not None else None
        self.unit = data['unit']
        self.header = fits.Header(data['header'])

        return

    def save(self, path, overwrite=False):
        """Pickles the map to a file.

        This method will fail if the map is associated to a Maps loaded
        from the db.

        Parameters:
            path (str):
                The path of the file to which the ``Map`` will be saved.
                Unlike for other Marvin Tools that derive from
                :class:`~marvin.core.core.MarvinToolsClass`, ``path`` is
                mandatory for ``Map`` given that the there is no default
                path for a given map.
            overwrite (bool):
                If True, and the ``path`` already exists, overwrites it.
                Otherwise it will fail.

        Returns:
            path (str):
                The realpath to which the file has been saved.

        """

        # check for file extension
        if not os.path.splitext(path)[1]:
            path = os.path.join(path + '.mpf')

        return marvin.core.marvin_pickle.save(self, path=path, overwrite=overwrite)

    @classmethod
    def restore(cls, path, delete=False):
        """Restores a Map object from a pickled file.

        If ``delete=True``, the pickled file will be removed after it has been
        unplickled. Note that, for map objes instantiated from a Maps object
        with ``data_origin='file'``, the original file must exists and be
        in the same path as when the object was first created.

        """

        return marvin.core.marvin_pickle.restore(path, delete=delete)

    def _make_image(self, snr_min, log_cb, use_mask):
        """Make masked array of image.

        Parameters:
            snr_min (float):
                Minimum signal-to-noise for keeping a valid measurement.
            log_cb (bool):
                If True, use log colorbar.
            use_mask (bool):
                If True, use DAP bitmasks.

        Returns:
            tuple: (masked array of image, tuple of (x, y) coordinates of bins with no measurement)
        """

        novalue = (self.mask & 2**4) > 0
        if use_mask:
            badvalue = (self.mask & 2**5) > 0
            matherror = (self.mask & 2**6) > 0
            badfit = (self.mask & 2**7) > 0
            donotuse = (self.mask & 2**30) > 0
            no_data = np.logical_or.reduce((novalue, badvalue, matherror, badfit, donotuse))

            if self.ivar is not None:
                # Flag a region as having no data if ivar = 0
                ivar_zero = (self.ivar == 0.)
                no_data = np.logical_or.reduce((no_data, ivar_zero))
        else:
            no_data = novalue

        no_measure = self._make_mask_no_measurement(self.value, self.ivar, snr_min, log_cb)

        no_data_no_measure = np.logical_or(no_data, no_measure)

        image = np.ma.array(self.value, mask=no_data_no_measure)
        mask_nodata = np.ma.array(np.ones(self.value.shape), mask=np.logical_not(no_data))

        return image, mask_nodata

    def _make_mask_no_measurement(self, data, ivar, snr_min, log_cb):
        """Mask invalid measurements within a data array.

        Parameters:
            data (array):
                Values for image.
            ivar (array):
                Inverse variance of values.
            snr_min (float):
                Minimum signal-to-noise for keeping a valid measurement.
            log_cb (bool):
                 If True, use log colorbar.

        Returns:
            array: Boolean array for mask (i.e., True corresponds to value to be masked out).
        """
        no_measure = np.zeros(data.shape, dtype=bool)

        if ivar is not None:
            if not np.all(np.isnan(ivar)):
                no_measure = (ivar == 0.)
                if snr_min is not None:
                    no_measure[(np.abs(data * np.sqrt(ivar)) < snr_min)] = True
                if log_cb:
                    no_measure[data <= 0.] = True

        return no_measure

    def _set_extent(self, cube_size, sky_coords):
        """Set extent of map.

        Parameters:
            cube_size (tuple):
                Size of the cube in spaxels.
            sky_coords (bool):
                If True, use sky coordinates, otherwise use spaxel coordinates.

        Returns:
            array
        """

        if sky_coords:
            spaxel_size = 0.5  # arcsec
            extent = np.array([-(cube_size[0] * spaxel_size), (cube_size[0] * spaxel_size),
                               -(cube_size[1] * spaxel_size), (cube_size[1] * spaxel_size)])
        else:
            extent = np.array([0, cube_size[0], 0, cube_size[1]])

        return extent

    def _set_patch_style(self, extent, facecolor='#A8A8A8'):
        """Set default parameters for a patch.

        Parameters:
            extent (tuple):
                Extent of image (xmin, xmax, ymin, ymax).
            facecolor (str):
                Background color. Default is '#A8A8A8' (gray).

        Returns:
            dict
        """

        if int(mpl.__version__.split('.')[0]) > 1:
            mpl.rcParams['hatch.linewidth'] = 0.5
            mpl.rcParams['hatch.color'] = 'w'

        patch_kws = dict(xy=(extent[0] + 0.01, extent[2] + 0.01),
                         width=extent[1] - extent[0] - 0.02,
                         height=extent[3] - extent[2] - 0.02, hatch='xxxx', linewidth=0,
                         fill=True, facecolor=facecolor, edgecolor='w', zorder=0)

        return patch_kws

    def _ax_setup(self, sky_coords, fig=None, ax=None, facecolor='#A8A8A8'):
        """Basic axis setup for maps.

        Parameters:
            sky_coords (bool):
                If True, show plot in sky coordinates (i.e., arcsec), otherwise show in spaxel
                coordinates.
            fig (plt.figure object):
                Matplotlib plt.figure object. Use if creating subplot of a multi-panel plot.
                Default is None.
            ax (plt.figure axis object):
                Matplotlib plt.figure axis object. Use if creating subplot of a multi-panel plot.
                Default is None.
            facecolor (str):
                Axis facecolor. Default is '#A8A8A8'.

        Returns:
            tuple: (plt.figure object, plt.figure axis object)
        """

        xlabel = 'arcsec' if sky_coords else 'spaxel'
        ylabel = 'arcsec' if sky_coords else 'spaxel'

        if 'seaborn' in sys.modules:
            if ax is None:
                sns.set_context('poster', rc={'lines.linewidth': 2})
            else:
                sns.set_context('talk', rc={'lines.linewidth': 2})
            sns.set_style(rc={'axes.facecolor': facecolor})

        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.1, 2 / 3., 5 / 6.])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        if 'seaborn' not in sys.modules:
            if int(mpl.__version__.split('.')[0]) < 2:
                ax.set_axis_bgcolor(facecolor)
            else:
                ax.set_facecolor(facecolor)

        ax.grid(False, which='both', axis='both')

        return fig, ax

    def plot(self, *args, **kwargs):
        """Make single panel map or one panel of multi-panel map plot.

        Parameters:
            cmap (str):
                Default is ``RdBu_r`` for velocities, ``inferno`` for sigmas, and ``linear_Lab``
                for other properties.
            percentile_clip (tuple-like):
                Percentile clip. Default is ``(10, 90)`` for velocities and sigmas and ``(5, 95)``
                for other properties.
            sigma_clip (float):
                Sigma clip. Default is ``None``.
            cbrange (tuple-like):
                If ``None``, set automatically. Default is ``None``.
            symmetric (bool):
                Draw a colorbar that is symmetric around zero. Default is ``True`` for velocities
                and ``False`` for other properties.
            snr_min (float):
                Minimum signal-to-noise for keeping a valid measurement. Default is ``1``.
            log_cb (bool):
                Draw a log normalized colorbar. Default is ``False``.
            title (str):
                If ``None``, set automatically from property (and channel) name(s). For no title,
                set to ''. Default is ``None``.
            cblabel (str):
                If ``None``, set automatically from unit. For no colorbar label, set to ''. Default
                is ``None``.
            sky_coords (bool):
                If ``True``, show plot in sky coordinates (i.e., arcsec), otherwise show in spaxel
                coordinates. Default is ``False``.
            use_mask (bool): Use DAP bitmasks. Default is ``True``.
            fig (matplotlib Figure object):
                Use if creating subplot of a multi-panel plot. Default is ``None``.
            ax (matplotlib Axis object):
                Use if creating subplot of a multi-panel plot. Default is ``None``.
            imshow_kws (dict):
                Keyword args to pass to `ax.imshow
                <http://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow>`_.
                Default is ``None``.
            cb_kws (dict):
                Keyword args to set and draw colorbar. Default is ``None``.

        Returns:
            fig, ax (tuple):
                `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_,
                `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_

        Example:
            >>> maps = Maps(plateifu='8485-1901')
            >>> haflux = maps.getMap('emline_gflux', channel='ha_6564')
            >>> haflux.plot()
        """

        valid_kwargs = ['cmap', 'percentile_clip', 'sigma_clip', 'cbrange', 'symmetric', 'snr_min',
                        'log_cb', 'title', 'cblabel', 'sky_coords', 'use_mask', 'fig', 'ax',
                        'imshow_kws', 'cb_kws']

        assert len(args) == 0, 'Map.plot() does not accept arguments, only keywords.'
        for kw in kwargs:
            assert kw in valid_kwargs, 'keyword {0} is not valid'.format(kw)

        assert ((kwargs.get('percentile_clip', None) is not None) +
                (kwargs.get('sigma_clip', None) is not None) +
                (kwargs.get('cbrange', None) is not None) <= 1), \
            'Only set one of percentile_clip, sigma_clip, or cbrange!'

        sigma_clip = kwargs.get('sigma_clip', None)
        cbrange = kwargs.get('cbrange', None)
        snr_min = kwargs.get('snr_min', 1)
        log_cb = kwargs.get('log_cb', False)
        title = kwargs.get('title', None)
        cblabel = kwargs.get('cblabel', None)
        sky_coords = kwargs.get('sky_coords', False)
        use_mask = kwargs.get('use_mask', True)
        fig = kwargs.get('fig', None)
        ax = kwargs.get('ax', None)
        imshow_kws = kwargs.get('imshow_kws', {})
        cb_kws = kwargs.get('cb_kws', {})

        image, nodata = self._make_image(snr_min=snr_min, log_cb=log_cb, use_mask=use_mask)

        if title is None:
            title = self.property_name + ('' if self.channel is None else ' ' + self.channel)
            title = ' '.join(title.split('_'))

        if 'vel' in title:
            cmap = kwargs.get('cmap', 'RdBu_r')
            percentile_clip = kwargs.get('percentile_clip', [10, 90])
            symmetric = kwargs.get('symmetric', True)
        elif 'sigma' in title:
            cmap = kwargs.get('cmap', 'inferno')
            percentile_clip = kwargs.get('percentile_clip', [10, 90])
            symmetric = kwargs.get('symmetric', False)
        else:
            cmap = kwargs.get('cmap', 'linear_Lab')
            percentile_clip = kwargs.get('percentile_clip', [5, 95])
            symmetric = kwargs.get('symmetric', False)

        if sigma_clip is not None:
            percentile_clip = None

        cb_kws['cmap'] = cmap
        cb_kws['percentile_clip'] = percentile_clip
        cb_kws['sigma_clip'] = sigma_clip
        cb_kws['cbrange'] = cbrange
        cb_kws['symmetric'] = symmetric
        cb_kws['label'] = self.unit if cblabel is None else cblabel
        cb_kws = colorbar.set_cb_kws(cb_kws)
        cb_kws = colorbar.set_cbrange(image, cb_kws)

        extent = self._set_extent(self.value.shape, sky_coords)
        imshow_kws.setdefault('extent', extent)
        imshow_kws.setdefault('interpolation', 'nearest')
        imshow_kws.setdefault('origin', 'lower')
        imshow_kws['norm'] = LogNorm() if log_cb else None

        fig, ax = self._ax_setup(sky_coords=sky_coords, fig=fig, ax=ax)

        # Plot regions with no measurement as hatched by putting one large patch as lowest layer
        patch_kws = self._set_patch_style(extent=extent)
        ax.add_patch(mpl.patches.Rectangle(**patch_kws))

        # Plot regions of no data as a solid color (gray #A8A8A8)
        A8A8A8 = colorbar.one_color_cmap(color='#A8A8A8')
        ax.imshow(nodata, cmap=A8A8A8, zorder=1, **imshow_kws)

        imshow_kws = colorbar.set_vmin_vmax(imshow_kws, cb_kws['cbrange'])
        p = ax.imshow(image, cmap=cb_kws['cmap'], zorder=10, **imshow_kws)

        fig, cb = colorbar.draw_colorbar(fig, p, **cb_kws)

        if title is not '':
            ax.set_title(label=title)

        # turn on to preserve zorder when saving to pdf (or other vector based graphics format)
        mpl.rcParams['image.composite_image'] = False

        return fig, ax
