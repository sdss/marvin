#!/usr/bin/env python
# encoding: utf-8
#
# Licensed under a 3-clause BSD license.
#
#
# map.py
#
# Created by Brett Andrews on 28 Apr 2017.
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


from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import marvin.utils.plot.colorbar as colorbar


def no_coverage_mask(values, mask):
    """Make mask of spaxels that are not covered by the IFU.
    
    Parameters:
        values (array):
            Values for image.
        mask (array):
            Mask for values.

    Returns:
        array: Boolean array for mask (i.e., True corresponds to value to be masked out).
    """

    return (mask & 2**0).astype(bool)
    # TODO create this later
    # return np.ma.array(np.ones(values.shape), mask=nocov)

def bad_data_mask(values, mask):
    """Make mask of spaxels that are masked as bad data by the DAP.
    
    The masks that are considered bad data are "BADVALUE," "MATHERROR," "BADFIT," and "DONOTUSE,"
    which correspond to the DAPPIXEL bitmasks 5, 6, 7, and 30, respectively.
    
    Parameters:
        values (array):
            Values for image.
        mask (array):
            Mask for values.

    Returns:
        array: Boolean array for mask (i.e., True corresponds to value to be masked out).
    """
    badvalue = (mask & 2**5).astype(bool)
    matherror = (mask & 2**6).astype(bool)
    badfit = (mask & 2**7).astype(bool)
    donotuse = (mask & 2**30).astype(bool)
    return np.logical_or.reduce((badvalue, matherror, badfit, donotuse))

def low_snr_mask(values, ivar, snr_min, log_cb):
    """Mask spaxels with a signal-to-noise ratio lower than some threshold.

    Parameters:
        values (array):
            Values for image.
        ivar (array):
            Inverse variance of values.
        snr_min (float):
            Minimum signal-to-noise for keeping a valid measurement.

    Returns:
        array: Boolean array for mask (i.e., True corresponds to value to be masked out).
    """

    low_snr = np.zeros(values.shape, dtype=bool)

    if (ivar is not None) and (not np.all(np.isnan(ivar))):
        low_snr = (ivar == 0.)
        
        if snr_min is not None:
            low_snr[np.abs(values * np.sqrt(ivar)) < snr_min] = True
        


    return low_snr

def log_colorbar_mask(values, log_cb):
    """Mask spaxels with negative values when using a logarithmic colorbar.

    Parameters:
        values (array):
            Values for image.
        log_cb (bool):
            Use logarithmic colorbar.

    Returns:
        array: Boolean array for mask (i.e., True corresponds to value to be masked out).
    """

    mask = np.zeros(values.shape, dtype=bool)

    if log_cb:
        mask[values <= 0.] = True

    return mask

def _make_image(values, mask, snr_min, log_cb, use_mask):
    """Make masked array of image.

    Parameters:
        value (array):
            2D array of values.
        mask (array):
            2D bitmask array.
        snr_min (float):
            Minimum signal-to-noise for keeping a valid measurement.
        log_cb (bool):
            If True, use log colorbar.
        use_mask (bool):
            If True, use DAP bitmasks.

    Returns:
        tuple: (masked array of image, tuple of (x, y) coordinates of bins with no measurement)
    """

    # spaxels outside of the coverage of the IFU are gray
    mask_nocov = no_coverage_mask(values, mask)

    # hatch bad data (flagged and low SNR)
    if use_mask:
        badvalue = (self.mask & 2**5).astype(bool)
        matherror = (self.mask & 2**6).astype(bool)
        badfit = (self.mask & 2**7).astype(bool)
        donotuse = (self.mask & 2**30).astype(bool)
        low_snr = self._make_mask_low_snr(self.values, self.ivar, snr_min, log_cb)
        bad_data = np.logical_or.reduce((badvalue, matherror, badfit, donotuse, low_snr))

        # TODO can I remove this because it is redundant with _make_mask_low_snr()?
        # if self.ivar is not None:
        #     # Flag a region as having no data if ivar = 0
        #     ivar_zero = (self.ivar == 0.)
        #     bad_data = np.logical_or.reduce((bad_data, ivar_zero))
    else:
        bad_data = ~nocov  # set to None?

    image = np.ma.array(self.values, mask=np.logical_or(nocov, bad_data))

    return image, mask_nocov

def make_image(values, nocov, bad_data, low_snr):
    return np.ma.array(values, mask=np.logical_or(nocov, bad_data, low_snr))


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

        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.1, 2 / 3., 5 / 6.])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

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

        if title is None:
            title = self.property_name + ('' if self.channel is None else ' ' + self.channel)
            title = ' '.join(title.split('_'))

        if 'vel' in title:
            cmap = kwargs.get('cmap', 'RdBu_r')
            percentile_clip = kwargs.get('percentile_clip', [10, 90])
            symmetric = kwargs.get('symmetric', True)
            snr_min = None
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

        image, nodata = self._make_image(snr_min=snr_min, log_cb=log_cb, use_mask=use_mask)

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
