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
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from distutils import version
import os
import sys
import warnings

from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
from matplotlib.colors import LogNorm
from matplotlib.colors import from_levels_and_colors

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

#### from mangadap.plot import util
#### from mangadap.plot import colorbar


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

    def __repr__(self):

        return ('<Marvin Map (plateifu={0.maps.plateifu!r}, property={0.property_name!r}, '
                'channel={0.channel!r})>'.format(self))

    def _load_map_from_file(self):
        """Initialises de Map from a ``Maps`` with ``data_origin='file'``."""

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
        """Initialises de Map from a ``Maps`` with ``data_origin='db'``."""

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

    def _load_map_from_api(self):
        """Initialises de Map from a ``Maps`` with ``data_origin='api'``."""

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

    def plot(self, array='value', xlim=None, ylim=None, zlim=None,
             xlabel=None, ylabel=None, zlabel=None, cmap=None, kw_imshow=None,
             figure=None, return_figure=False, show_masked=False):
        """Plot a map using matplotlib.

        Returns a |axes|_ object with a representation of this map.
        The returned ``axes`` object can then be showed, modified, or saved to
        a file. If running Marvin from an iPython console and
        `matplotlib.pyplot.ion()
        <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.ion>`_,
        the plot will be displayed interactivelly.

        Parameters:
            array ({'value', 'ivar', 'mask'}):
                The array to display, either the data itself, the inverse
                variance, or the mask.
            xlim,ylim (tuple-like or None):
                The range to display for the x- and y-axis, respectively,
                defined as a tuple of two elements ``[xmin, xmax]``. If
                the range is ``None``, the range for the axis will be set
                automatically by matploltib.
            zlim (tuple or None):
                The range to display in the z-axis (intensity level). If
                ``None``, the default scaling provided by matplotlib will be
                used.
            xlabel,ylabel,zlabel (str or None):
                The axis labels to be passed to the plot.
            cmap (``matplotlib.pyplot.cm`` colourmap or None):
                The matplotlib colourmap to use (see
                `this <http://matplotlib.org/users/colormaps.html#list-colormaps>`_
                page for possible colourmaps). If ``None``, defaults to
                ``coolwarm_r``.
            kw_imshow (dict):
                Any other kwyword arguments to be passed to
                `imshow <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow>`_.
            figure (matplotlib Figure object or None):
                The matplotlib figure object from which the axes must be
                created. If ``figure=None``, a new figure will be created.
            return_figure (bool):
                If ``True``, the matplotlib Figure object used will be returned
                along with the axes object.
            show_masked (bool):
                By default, masked values are not shown.
                If ``show_masked=True``, all spaxels are shown.

        Returns:
            ax (`matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_):
                The matplotlib axes object containing the plot representing
                the map. If ``return_figure=True``, a tuple will be
                returned of the form ``(ax, fig)``.

        Example:

          >>> maps = Maps(plateifu='8485-1901')
          >>> ha_map = maps.getMap('emline_gflux', channel='ha_6564')
          >>> ha_map.plot()

        .. |axes| replace:: matplotlib.axes
        .. _axes: http://matplotlib.org/api/axes_api.html

        """

        # TODO: plot in sky coordinates. (JSG)

        array = array.lower()
        validExensions = ['value', 'ivar', 'mask']
        assert array in validExensions, 'array must be one of {0!r}'.format(validExensions)

        if array == 'value':
            data = self.value
        elif array == 'ivar':
            data = self.ivar
        elif array == 'mask':
            data = self.mask

        fig = plt.figure() if figure is None else figure
        ax = fig.add_subplot(111)

        if zlim is not None:
            assert len(zlim) == 2
            vmin = zlim[0]
            vmax = zlim[1]
        else:
            vmin = None
            vmax = None

        if kw_imshow is None:
            kw_imshow = dict(vmin=vmin, vmax=vmax,
                             origin='lower', aspect='auto',
                             interpolation='none')

        if cmap is None:
            cmap = plt.cm.coolwarm_r

        if show_masked is False and array != 'mask':
            data = np.ma.array(data, mask=(self.mask > 0))

        imPlot = ax.imshow(data, cmap=cmap, **kw_imshow)

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)

        cBar = plt.colorbar(imPlot, cax=cax)
        cBar.solids.set_edgecolor('face')

        if xlim is not None:
            assert len(xlim) == 2
            ax.set_xlim(*xlim)

        if ylim is not None:
            assert len(ylim) == 2
            ax.set_ylim(*ylim)

        if xlabel is None:
            xlabel = 'x [pixels]'

        if ylabel is None:
            ylabel = 'y [pixels]'

        if zlabel is None:
            zlabel = r'{0}'.format(self.unit)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cBar.set_label(zlabel)

        if return_figure:
            return (ax, fig)
        else:
            return ax

# ======================================================================

    def _make_image(self, data, snr_thresh, log_colorbar):
        """Make masked array of image.

        Args:
            data (array): Data.
            snr_thresh (float): Signal-to-noise theshold for keeping a valid measurement.
            log_colorbar (bool): If True, use log colorbar.

        Returns:
            tuple: (masked array of image,
                    tuple of (x, y) coordinates of bins with no measurement)
        """

        # Flag a region as having no data if ivar = 0
        ivar_zero = (self.ivar == 0.)

        novalue = (self.mask & 2**4) > 0
        badvalue = (self.mask & 2**5) > 0
        matherror = (self.mask & 2**6) > 0
        badfit = (self.mask & 2**7) > 0
        donotuse = (self.mask & 2**30) > 0
        no_data = np.logical_or.reduce((ivar_zero, novalue, badvalue, matherror, badfit, donotuse))

        no_measure = self._make_mask_no_measurement(data, self.ivar, snr_thresh, log_colorbar)

        no_data_no_measure = np.logical_or(no_data, no_measure)

        image = np.ma.array(data, mask=no_data_no_measure)
        mask_nodata = np.ma.array(np.ones(data.shape), mask=np.logical_not(no_data))

        return image, mask_nodata

    def _make_mask_no_measurement(self, data, ivar, snr_thresh, log_colorbar):
        """Mask invalid measurements within a data array.

        Args:
            data (array): Data.
            ivar (array): Inverse variance.
            snr_thresh (float): Signal-to-noise threshold for keeping a valid measurement.
            log_colorbar (bool): If True, use log colorbar.

        Returns:
            array: Boolean array for mask (i.e., True corresponds to value to be
                masked out).
        """

        if np.all(np.isnan(ivar)):
            ivar = None

        if ivar is not None:
            no_measure = (ivar == 0.)
            if snr_thresh is not None:
                no_measure[(np.abs(data * np.sqrt(ivar)) < snr_thresh)] = True
            if log_colorbar:
                no_measure[data <= 0.] = True

        return no_measure

    def _set_extent(self, cube_size, sky_coords):
        """Set extent of map.

        Args:
            cube_size (tuple): Size of the cube in spaxels.
            sky_coords (bool): If True, use sky coordinates, otherwise use spaxel coordinates.

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

    def _set_patch_style(self, extent, color='#A8A8A8'):
        """Set default parameters for a patch.

        Args:
            extent (tuple): Extent of image (xmin, xmax, ymin, ymax).
            color (str): Background color. Default is '#A8A8A8' (gray).

        Returns:
            dict
        """

        # TODO test this with matplotlib 2.0
        # if int(matplotlib.__version__.split('.')[0]) > 1:
        #     matplotlib.rcParams['hatch_linewidth'] = 0.1

        patch_kws = dict(xy=(extent[0] + 0.01, extent[2] + 0.01),
                         width=extent[1] - extent[0] - 0.02,
                         height=extent[3] - extent[2] - 0.02, hatch='xxxx', linewidth=0,
                         fill=True, fc=color, ec='w', zorder=0)

        return patch_kws

    def _ax_setup(self, sky_coords, fig=None, ax=None, fig_kws=None, facecolor='#EAEAF2'):
        """Basic axis setup for maps.

        Args:
            sky_coords (bool): If True, show plot in sky coordinates (i.e., arcsec), otherwise show
                in spaxel coordinates.
            fig: Matplotlib plt.figure object. Use if creating subplot of a multi-panel plot.
                Default is None.
            ax: Matplotlib plt.figure axis object. Use if creating subplot of a multi-panel
                plot. Default is None.
            fig_kws (dict): Keyword args to pass to plt.figure. Default is None.
            facecolor (str): Axis facecolor. Default is '#EAEAF2'.

        Returns:
            tuple: (plt.figure object, plt.figure axis object)
        """

        fig_kws = fig_kws or {}

        xlabel = 'arcsec' if sky_coords else 'spaxel'
        ylabel = 'arcsec' if sky_coords else 'spaxel'

        if 'seaborn' in sys.modules:
            if ax is None:
                sns.set_context('poster', rc={'lines.linewidth': 2})
            else:
                sns.set_context('talk', rc={'lines.linewidth': 2})
            sns.set_style(rc={'axes.facecolor': facecolor})

        if ax is None:
            fig = plt.figure(**fig_kws)
            ax = fig.add_axes([0.12, 0.1, 2/3., 5/6.])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        if 'seaborn' not in sys.modules:
            ax.set_axis_bgcolor(facecolor)

        ax.grid(False, which='both', axis='both')

        return fig, ax

    def dapplot(self, ext='value', snr_thresh=1, sky_coords=False, fig=None, ax=None, fig_kws=None,
                ax_kws=None, title_kws=None, patch_kws=None, imshow_kws=None, cb_kws=None):
        """Make single panel map or one panel of multi-panel map plot.

        Args:
            ext (str):  The array to display, either the data itself ('value'), the inverse
                variance ('ivar'), or the mask ('mask'). Default is 'value'.
            snr_thresh (float): Signal-to-noise threshold for keeping a valid measurement. Default
                is 1.
            sky_coords (bool): If True, show plot in sky coordinates (i.e., arcsec), otherwise show
                in spaxel coordinates. Default is False.
            fig: plt.figure object. Use if creating subplot of a multi-panel plot. Default is
                None.
            ax: plt.figure axis object. Use if creating subplot of a multi-panel plot. Default
                is None.
            fig_kws (dict): Keyword args to pass to plt.figure. Default is None.
            ax_kws (dict): Keyword args to draw axis. Default is None.
            title_kws (dict): Keyword args to pass to ax.set_title. Default is None.
            patch_kws (dict): Keyword args to pass to ax.add_patch. Default is None.
            imshow_kws (dict): Keyword args to pass to ax.imshow. Default is None.
            cb_kws (dict): Keyword args to set and draw colorbar. Default is None.

        Returns:
            tuple: (plt.figure object, plt.figure axis object)
        """

        fig_kws = fig_kws or {}
        ax_kws = ax_kws or {}
        title_kws = title_kws or {}
        patch_kws = patch_kws or {}
        imshow_kws = imshow_kws or {}
        cb_kws = cb_kws or {}

        ax_kws.setdefault('facecolor', '#A8A8A8')
        patch_kws.setdefault('facecolor', '#A8A8A8')

        ext = ext.lower()
        validExtensions = ['value', 'ivar', 'mask']
        assert ext in validExtensions, 'ext must be one of {0!r}'.format(validExtensions)

        if ext == 'value':
            data = self.value
        elif ext == 'ivar':
            data = self.ivar
        elif ext == 'mask':
            data = self.mask

        fig, ax = self._ax_setup(sky_coords=sky_coords, fig=fig, ax=ax, fig_kws=fig_kws, **ax_kws)

        if title_kws.get('label', None) is not None:
            ax.set_title(**title_kws)

        cb_kws = colorbar.set_cb_kws(cb_kws)

        extent = self._set_extent(data.shape, sky_coords)

        imshow_kws['extent'] = extent
        imshow_kws['cmap'] = cb_kws['cmap']
        if cb_kws.get('log_colorbar', False):
            imshow_kws['norm'] = LogNorm()

        patch_kws = self._set_patch_style(extent=extent)

        image, nodata = self._make_image(data, snr_thresh=snr_thresh,
                                         log_colorbar=cb_kws.get('log_colorbar', False))
        cb_kws = colorbar.set_cbrange(image, cb_kws)
        imshow_kws = colorbar.set_vmin_vmax(imshow_kws, cb_kws['cbrange'])

        # Plot regions with no measurement as hatched by putting one large patch as lowest layer
        ax.add_patch(matplotlib.patches.Rectangle(**patch_kws))

        # Plot regions of no data as a solid color (gray #A8A8A8)
        ax.imshow(nodata, interpolation='none', origin='lower', extent=imshow_kws['extent'],
                  cmap=colorbar.one_color_cmap(color='#A8A8A8'), zorder=1)

        p = ax.imshow(image, interpolation='none', origin='lower', **imshow_kws)

        fig, cb = colorbar.draw_colorbar(fig, p, **cb_kws)

        # turn on to preserve zorder when saving to pdf (or other vector based graphics format)
        matplotlib.rcParams['image.composite_image'] = False

        if 'seaborn' in sys.modules:
            sns.set_style(rc={'axes.facecolor': '#EAEAF2'})

        return fig, ax
