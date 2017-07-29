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
# Includes code from mangadap.plot.maps.py licensed under the following 3-clause
# BSD license.
#
# Copyright (c) 2015, SDSS-IV/MaNGA Pipeline Group
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
# HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import division, print_function, absolute_import

import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from marvin import config
import marvin.utils.plot.colorbar as colorbar
from marvin.utils.general import get_plot_params


def no_coverage_mask(mask, bit, ivar=None):
    """Mask spaxels that are not covered by the IFU.

    Parameters:
        mask (array):
            Mask for value.
        bit (int):
            Bit for "NOCOV."
        ivar (array):
            Inverse variance for image. Default is None.

    Returns:
        array: Boolean array for mask (i.e., True corresponds to value to be
        masked out).
    """
    assert (bit is not None) or (ivar is not None), 'Must provide a bit or ivar array.'

    return (mask & 2**bit).astype(bool) if bit is not None else (ivar == 0)


def bad_data_mask(mask, bits):
    """Mask spaxels that are flagged as bad data by the DAP.

    The masks that are considered bad data are "UNRELIABLE" and "DONOTUSE."
    Note: MPL-4 used only a good = 0 and bad = 1 mask. The "bad" flag
    corresponds most closely to "DONOTUSE."

    Parameters:
        mask (array):
            Mask for value.
        bits (dict):
            Bits that indicate bad data.

    Returns:
        array: Boolean array for mask (i.e., True corresponds to value to be
        masked out).
    """
    if 'unreliable' in bits.keys():     
        unreliable = (mask & 2**bits['unreliable']).astype(bool)
    else:
        unreliable = np.zeros(mask.shape, dtype=bool)

    donotuse = (mask & 2**bits['doNotUse']).astype(bool)
    return np.logical_or.reduce((unreliable, donotuse))


def low_snr_mask(value, ivar, snr_min):
    """Mask spaxels with a signal-to-noise ratio below some threshold.

    Parameters:
        value (array):
            Value for image.
        ivar (array):
            Inverse variance of value.
        snr_min (float):
            Minimum signal-to-noise for keeping a valid measurement.

    Returns:
        array: Boolean array for mask (i.e., True corresponds to value to be
        masked out).
    """
    low_snr = np.zeros(value.shape, dtype=bool)

    if (ivar is not None) and (not np.all(np.isnan(ivar))):
        low_snr = (ivar == 0.)

        if snr_min is not None:
            low_snr[np.abs(value * np.sqrt(ivar)) < snr_min] = True

    return low_snr


def log_colorbar_mask(value, log_cb):
    """Mask spaxels with negative value when using logarithmic colorbar.

    Parameters:
        value (array):
            Value for image.
        log_cb (bool):
            Use logarithmic colorbar.

    Returns:
        array: Boolean array for mask (i.e., True corresponds to value to be
        masked out).
    """
    mask = np.zeros(value.shape, dtype=bool)

    if log_cb:
        mask[value <= 0.] = True

    return mask

def _get_prop(title):
    """Gets property name from plot title.
    
    Parameters:
        title (str):
            Plot title.
    
    Returns:
        str
    """
    if 'vel' in title:
        return 'vel'
    elif 'sigma' in title:
        return 'sigma'
    else:
        return 'default'

def select_good_spaxels(value, nocov, bad_data, low_snr, log_cb_mask):
    """Create masked array of spaxels to display.

    Parameters:
        value (array):
            Value for image.
        nocov (array):
            Mask for spaxels without IFU coverage.
        bad_data (array):
            Mask for data flagged as bad (see
            :ref:`marvin-utils-plot-map-default-params` for default bits).
        low_snr (array):
            Mask for data below the signal-to-noise ratio threshold.
        low_cb_mask (array):
            Mask for negative elements of ``value`` if using a logarithmic
            colorbar.

    Returns:
        masked array: spaxels to display in plot.
    """
    return np.ma.array(value, mask=np.logical_or.reduce((nocov, bad_data, low_snr, log_cb_mask)))


def set_extent(cube_size, sky_coords):
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
        extent = np.array([0, cube_size[0] - 1, 0, cube_size[1] - 1])

    return extent


def set_patch_style(extent, facecolor='#A8A8A8'):
    """Set default parameters for a patch.

    Parameters:
        extent (tuple):
            Extent of image (xmin, xmax, ymin, ymax).
        facecolor (str):
            Background color. Default is '#A8A8A8' (gray).

    Returns:
        dict
    """
    patch_kws = dict(xy=(extent[0] + 0.01, extent[2] + 0.01),
                     width=extent[1] - extent[0] - 0.02,
                     height=extent[3] - extent[2] - 0.02, hatch='xxxx', linewidth=0,
                     fill=True, facecolor=facecolor, edgecolor='w', zorder=0)

    return patch_kws


def ax_setup(sky_coords, fig=None, ax=None, facecolor='#A8A8A8'):
    """Do basic axis setup for maps.

    Parameters:
        sky_coords (bool):
            If True, show plot in sky coordinates (i.e., arcsec), otherwise
            show in spaxel coordinates.
        fig (plt.figure object):
            Matplotlib plt.figure object. Use if creating subplot of a
            multi-panel plot. Default is None.
        ax (plt.figure axis object):
            Matplotlib plt.figure axis object. Use if creating subplot of a
            multi-panel plot. Default is None.
        facecolor (str):
            Axis facecolor. Default is '#A8A8A8'.

    Returns:
        tuple: (plt.figure object, plt.figure axis object)
    """
    xlabel = 'arcsec' if sky_coords else 'spaxel'
    ylabel = 'arcsec' if sky_coords else 'spaxel'

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if int(mpl.__version__.split('.')[0]) <= 1:
        ax.set_axis_bgcolor(facecolor)
    else:
        ax.set_facecolor(facecolor)
        ax.grid(False, which='both', axis='both')

    return fig, ax


def set_title(title=None, property_name=None, channel=None):
    """Set title for map.

    Parameters:
        title (str):
            If ``None``, try to set automatically from property (and channel)
            name(s). For no title, set to ''. Default is ``None``.
        property_str (str):
            Map property name. Default is ``None``.
        channel (str):
            Map channel name. Default is ``None``.
    Returns:
        str
    """
    if title is None:
        property_name = property_name if property_name is not None else ''
        channel = channel if channel is not None else ''
        title = ' '.join((property_name, channel))
        title = ' '.join(title.split('_')).strip()

    return title


def plot(*args, **kwargs):
    """Make single panel map or one panel of multi-panel map plot.

    Parameters:
        dapmap (marvin.tools.map.Map):
            Marvin Map object. Default is ``None``.
        value (array):
            Data array. Default is ``None``.
        ivar (array):
            Inverse variance array. Default is ``None``.
        mask (array):
            Mask array. Default is ``None``.
        cmap (str):
            Colormap (see :ref:`marvin-utils-plot-map-default-params` for
            defaults).
        percentile_clip (tuple-like):
            Percentile clip (see :ref:`marvin-utils-plot-map-default-params`
            for defaults).
        sigma_clip (float):
            Sigma clip. Default is ``False``.
        cbrange (tuple-like):
            If ``None``, set automatically. Default is ``None``.
        symmetric (bool):
            Draw a colorbar that is symmetric around zero (see
            :ref:`marvin-utils-plot-map-default-params` for default).
        snr_min (float):
            Minimum signal-to-noise for keeping a valid measurement (see
            :ref:`marvin-utils-plot-map-default-params` for default).
        log_cb (bool):
            Draw a log normalized colorbar. Default is ``False``.
        title (str):
            If ``None``, set automatically from property (and channel) name(s).
            For no title, set to ''. Default is ``None``.
        cblabel (str):
            If ``None``, set automatically from unit. For no colorbar label,
            set to ''. Default is ``None``.
        sky_coords (bool):
            If ``True``, show plot in sky coordinates (i.e., arcsec), otherwise
            show in spaxel coordinates. Default is ``False``.
        use_mask (bool):
            Use DAP bitmasks. Default is ``True``.
        plt_style (str):
            Matplotlib style sheet to use. Default is 'seaborn-darkgrid'.
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
        return_cb (bool):
            Return colorbar axis. Default it ``False``.

    Returns:
        fig, ax (tuple):
            `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_,
            `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_

    Example:
        >>> import marvin.utils.plot.map as mapplot
        >>> maps = Maps(plateifu='8485-1901')
        >>> ha = maps['emline_gflux_ha_6564']
        >>> fig, ax = mapplot.plot(dapmap=ha)
    """
    valid_kwargs = ['dapmap', 'value', 'ivar', 'mask', 'cmap', 'percentile_clip', 'sigma_clip',
                    'cbrange', 'symmetric', 'snr_min', 'log_cb', 'title', 'cblabel', 'sky_coords',
                    'use_mask', 'plt_style', 'fig', 'ax', 'imshow_kws', 'cb_kws', 'return_cb']

    assert len(args) == 0, 'Map.plot() does not accept arguments, only keywords.'

    for kw in kwargs:
        assert kw in valid_kwargs, 'keyword {0} is not valid'.format(kw)

    assert ((kwargs.get('percentile_clip', False)) +
            (kwargs.get('sigma_clip', False)) +
            (kwargs.get('cbrange', None) is not None) <= 1), \
        'Only set one of percentile_clip, sigma_clip, or cbrange!'

    dapmap = kwargs.get('dapmap', None)
    value = kwargs.get('value', None)
    ivar = kwargs.get('ivar', None)
    mask = kwargs.get('mask', None)
    sigma_clip = kwargs.get('sigma_clip', False)
    cbrange = kwargs.get('cbrange', None)
    log_cb = kwargs.get('log_cb', False)
    title = kwargs.get('title', None)
    cblabel = kwargs.get('cblabel', None)
    sky_coords = kwargs.get('sky_coords', False)
    use_mask = kwargs.get('use_mask', True)
    plt_style = kwargs.get('plt_style', 'seaborn-darkgrid')
    fig = kwargs.get('fig', None)
    ax = kwargs.get('ax', None)
    imshow_kws = kwargs.get('imshow_kws', {})
    cb_kws = kwargs.get('cb_kws', {})
    return_cb = kwargs.get('return_cb', False)

    assert (value is not None) or (dapmap is not None), \
        'Map.plot() requires specifying ``value`` or ``dapmap``.'

    # user-defined value, ivar, or mask overrides dapmap attributes
    value = value if value is not None else getattr(dapmap, 'value', None)
    ivar = ivar if ivar is not None else getattr(dapmap, 'ivar', None)
    mask = mask if mask is not None else getattr(dapmap, 'mask', np.zeros(value.shape, dtype=bool))

    title = set_title(title,
                      property_name=getattr(dapmap, 'property_name', None),
                      channel=getattr(dapmap, 'channel', None))

    # get plotparams from datamodel
    dapver = config.lookUpVersions()[1]
    prop = _get_prop(title)
    params = get_plot_params(dapver, prop)
    cmap = kwargs.get('cmap', params['cmap'])
    percentile_clip = kwargs.get('percentile_clip', params['percentile_clip'])
    symmetric = kwargs.get('symmetric', params['symmetric'])
    snr_min = kwargs.get('snr_min', params['snr_min'])

    if sigma_clip:
        percentile_clip = False

    # create no coverage, bad data, low SNR, and log colorbar masks
    nocov_mask = no_coverage_mask(mask, params['bitmasks'].get('nocov', None), ivar)
    badData = params['bitmasks']['badData']
    bad_data = bad_data_mask(mask, badData) if use_mask else np.zeros(value.shape)
    low_snr = low_snr_mask(value, ivar, snr_min) if use_mask else np.zeros(value.shape)
    log_cb_mask = log_colorbar_mask(value, log_cb)

    # final masked array to show
    good_spax = select_good_spaxels(value, nocov_mask, bad_data, low_snr, log_cb_mask)

    # setup colorbar
    cb_kws['cmap'] = cmap
    cb_kws['percentile_clip'] = percentile_clip
    cb_kws['sigma_clip'] = sigma_clip
    cb_kws['cbrange'] = cbrange
    cb_kws['symmetric'] = symmetric
    cb_kws['label'] = cblabel if cblabel is not None else getattr(dapmap, 'unit', '')
    cb_kws['log_cb'] = log_cb
    cb_kws = colorbar._set_cb_kws(cb_kws)
    cb_kws = colorbar._set_cbrange(good_spax, cb_kws)

    # setup unmasked spaxels
    extent = set_extent(value.shape, sky_coords)
    imshow_kws.setdefault('extent', extent)
    imshow_kws.setdefault('interpolation', 'nearest')
    imshow_kws.setdefault('origin', 'lower')
    imshow_kws['norm'] = LogNorm() if log_cb else None

    # setup background
    nocov_kws = copy.deepcopy(imshow_kws)
    nocov = np.ma.array(np.ones(value.shape), mask=~nocov_mask)
    A8A8A8 = colorbar._one_color_cmap(color='#A8A8A8')

    # setup masked spaxels
    patch_kws = set_patch_style(extent=extent)

    # finish setup of unmasked spaxels and colorbar range
    imshow_kws = colorbar._set_vmin_vmax(imshow_kws, cb_kws['cbrange'])

    # set hatch color and linewidths (in matplotlib 2.0+)
    try:
        mpl_rc = {it: mpl.rcParams[it] for it in ['hatch.color', 'hatch.linewidth']}
        mpl.rc_context({'hatch.color': 'w', 'hatch.linewidth': '0.5'})
    except KeyError as ee:
        mpl_rc = {}

    with plt.style.context(plt_style):

        fig, ax = ax_setup(sky_coords=sky_coords, fig=fig, ax=ax)

        # plot hatched regions by putting one large patch as lowest layer
        # hatched regions are bad data, low SNR, or negative values if the colorbar is logarithmic
        ax.add_patch(mpl.patches.Rectangle(**patch_kws))

        # plot regions without IFU coverage as a solid color (gray #A8A8A8)
        ax.imshow(nocov, cmap=A8A8A8, zorder=1, **nocov_kws)

        # plot unmasked spaxels
        p = ax.imshow(good_spax, cmap=cb_kws['cmap'], zorder=10, **imshow_kws)

        fig, cb = colorbar._draw_colorbar(fig, mappable=p, ax=ax, **cb_kws)

        if title is not '':
            ax.set_title(label=title)

    # restore previous matplotlib rc parameters (as of matplotlib 2.0.2 this
    # redraws the hatches with the original rcParam settings)
    # mpl.rc_context(mpl_rc)

    # turn on to preserve zorder when saving to pdf (or other vector based graphics format)
    mpl.rcParams['image.composite_image'] = False

    output = (fig, ax) if not return_cb else (fig, ax, cb)
    return output
