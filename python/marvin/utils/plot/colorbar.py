#!/usr/bin/env python
# encoding: utf-8
#
# Licensed under a 3-clause BSD license.
#
# Original code from mangadap.plot.colorbar.py licensed under the following
# 3-clause BSD license.
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
#
#
# colorbar.py
#
# Created by Brett Andrews on 07 Jun 2016.
#
# Modified by Brett Andrews on 4 May 2017.

"""Functions for colorbars."""

from __future__ import (division, print_function, absolute_import, unicode_literals)

import os
from os.path import join

import numpy as np
import scipy.interpolate as interpolate

import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.colors import from_levels_and_colors

from astropy.stats import sigma_clip

import marvin
from marvin.core.exceptions import MarvinError


def _log_cbticks(cbrange):
    """Set ticks and ticklabels for a log normalized colorbar.

    Parameters:
        cbrange (list):
            Colorbar range.

    Returns:
        array
    """
    subs = [1., 2., 3., 6.]
    bottom = np.floor(np.log10(cbrange[0]))
    top = np.ceil(np.log10(cbrange[1]))
    decs = np.arange(bottom, top + 1)
    tmp = np.array([sub * 10.**dec for dec in decs for sub in subs])
    return tmp[np.logical_and((tmp >= cbrange[0]), (tmp <= cbrange[1]))]


def _log_tick_format(value):
    """Format tick labels for log axis.

    If value between ___, return as ___:
       (0, 999], int
       [0.1, 0.99], 1 digit float
       otherwise: exponential notation

    Parameters
        value (float):
            Input value.

    Returns:
        str
    """
    exp = np.floor(np.log10(value))
    base = value / 10**exp
    if exp in [0, 1, 2]:
        return '{0:d}'.format(int(value))
    elif exp == -1:
        return '{0:.1f}'.format(value)
    else:
        return '{0:d}e{1:d}'.format(int(base), int(exp))


def _set_vmin_vmax(d, cbrange):
    """Set minimum and maximum values of the color map."""
    if 'vmin' not in d.keys():
        d['vmin'] = cbrange[0]
    if 'vmax' not in d.keys():
        d['vmax'] = cbrange[1]
    return d


def _cbrange_sigma_clip(image, sigma):
    """Sigma clip colorbar range.

    Parameters:
        image (masked array):
            Image.
        sigma (float):
            Sigma to clip.

    Returns:
        list: Colorbar range.
    """
    try:
        imclip = sigma_clip(image.data[~image.mask], sigma=sigma)
    except TypeError:
        imclip = sigma_clip(image.data[~image.mask], sig=sigma)

    try:
        cbrange = [imclip.min(), imclip.max()]
    except ValueError:
        cbrange = [image.min(), image.max()]

    return cbrange


def _cbrange_percentile_clip(image, lower, upper):
    """Clip colorbar range according to percentiles.

    Parameters:
        image (masked array):
            Image.
        lower (float):
            Lower percentile boundary.
        upper (float):
            Upper percentile boundary.

    Returns:
        list: Colorbar range.
    """
    cblow = np.percentile(image.data[~image.mask], lower)
    cbup = np.percentile(image.data[~image.mask], upper)
    return [cblow, cbup]


def _cbrange_user_defined(cbrange, cbrange_user):
    """Set user-specified colorbar range.

    Parameters:
        cbrange (list):
            Input colorbar range.
        cbrange_user (list):
            User-specified colorbar range. If a value is None, then use the
            previous value.

    Returns:
        list: Colorbar range.
    """
    for i in range(2):
        if cbrange_user[i] is not None:
            cbrange[i] = cbrange_user[i]
    return cbrange


def _set_cbrange(image, cb_kws):
    """Set colorbar range.

    Parameters:
        image (masked array):
            Image.
        cb_kws (dict):
            Colorbar kwargs.

    Returns:
        dict: Colorbar kwargs.
    """
    if cb_kws.get('sigma_clip'):
        cbr = _cbrange_sigma_clip(image, cb_kws['sigma_clip'])
    elif cb_kws.get('percentile_clip'):
        try:
            cbr = _cbrange_percentile_clip(image, *cb_kws['percentile_clip'])
        except IndexError:
            cbr = [0.1, 1]
    else:
        cbr = [image.min(), image.max()]

    if cb_kws.get('cbrange') is not None:
        cbr = _cbrange_user_defined(cbr, cb_kws['cbrange'])

    if cb_kws.get('symmetric', False):
        cb_max = np.max(np.abs(cbr))
        cbr = [-cb_max, cb_max]

    cbr, cb_kws['ticks'] = _set_cbticks(cbr, cb_kws)

    if cb_kws.get('log_cb', False):
        try:
            im_min = np.min(image[image > 0.])
        except ValueError:
            im_min = 0.1
        if im_min is np.ma.masked:
            im_min = 0.1
        cbr[0] = np.max((cbr[0], im_min))

    cb_kws['cbrange'] = cbr

    return cb_kws


def _set_cbticks(cbrange, cb_kws):
    """Set colorbar ticks.

    Adjust colorbar range if using a discrete colorbar so that the ticks fall
        in the middle of each level.

    Parameters:
        cbrange (list):
            Colorbar range.
        cb_kws (dict):
            Keyword args to set and draw colorbar.

    Return:
        tuple: colorbar range, colorbar tick numbers
    """
    if cb_kws.get('log_cb'):
        ticks = _log_cbticks(cbrange)
    else:
        try:
            ticks = MaxNLocator(cb_kws.get('n_ticks', 7)).tick_values(*cbrange)
        except AttributeError:
            print('AttributeError: MaxNLocator instance has no attribute ``tick_values``.')

    # if discrete colorbar, offset upper and lower cbrange so ticks are in center of each level
    if cb_kws.get('n_levels', None) is not None:
        offset = (ticks[1] - ticks[0]) / 2.
        cbrange = [ticks[0] - offset, ticks[-1] + offset]
        if cb_kws.get('tick_everyother', False):
            ticks = ticks[::2]

    return cbrange, ticks


def _draw_colorbar(fig, mappable, ax=None, axloc=None, cbrange=None, ticks=None, log_cb=False,
                  label_kws=None, tick_params_kws=None, **extras):
    """Make colorbar.

    Parameters:
        fig:
            `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_
            object from which the axes must be created.
        mappable (matplotlib image object):
            Matplotlib plotting element to map to colorbar.
        ax:
            `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_ object
            from which to steal space.
        axloc (list):
            Specify (left, bottom, width, height) of colorbar axis. Default is
            ``None``.
        cbrange (list):
            Colorbar min and max.
        ticks (list):
            Ticks on colorbar.
        log_cb (bool):
            Log colorbar. Default is ``False``.
        label_kws (dict):
            Keyword args to set colorbar label. Default is ``None``.
        tick_params_kws (dict):
            Keyword args to set colorbar tick parameters. Default is ``None``.

    Returns:
        fig, ax (tuple):
            `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_,
            `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_
    """
    label_kws = label_kws or {}
    tick_params_kws = tick_params_kws or {}

    cax = (fig.add_axes(axloc) if axloc is not None else None)
    try:
        cb = fig.colorbar(mappable=mappable, cax=cax, ax=ax, ticks=ticks)
    except ValueError:
        cb = None
    else:
        cb.ax.tick_params(**tick_params_kws)
        if label_kws.get('label') is not None:
            cb.set_label(**label_kws)
        if log_cb:
            cb.set_ticklabels([_log_tick_format(tick) for tick in ticks])

    return fig, cb


def _set_cmap(cm_name, n_levels=None):
    """Set the colormaps.

    Parameters:
        cm_name (str):
            Name of colormap.
        n_levels (int):
            Number of discrete levels of colormap. If ``None``, then produce
            continuous colormap. Default is ``None``.

    Returns:
        `matplotlib.cm <http://matplotlib.org/api/cm_api.html>`_ (colormap) object
    """
    cmap = _string_to_cmap(cm_name)

    if n_levels is not None:
        cmap = _cmap_discretize(cmap, n_levels)

    return cmap


def _string_to_cmap(cm_name):
    """Return colormap given name.

    Parameters:
        cm_name (str):
            Name of colormap.

    Returns:
        `matplotlib.cm <http://matplotlib.org/api/cm_api.html>`_ (colormap)
        object
    """
    if isinstance(cm_name, str):
        if 'linearlab' in cm_name:
            try:
                cmap, cmap_r = linearlab()
            except IOError:
                cmap = cm.viridis
            else:
                if '_r' in cm_name:
                    cmap = cmap_r
        else:
            cmap = cm.get_cmap(cm_name)
    elif isinstance(cm_name, ListedColormap) or isinstance(cm_name, LinearSegmentedColormap):
        cmap = cm_name
    else:
        raise MarvinError('{} is not a valid cmap'.format(cm_name))

    return cmap


def _set_cb_kws(cb_kws):
    """Set colorbar keyword args.

    Parameters:
        cb_kws (dict):
            Colorbar keyword args.

    Returns:
        dict
    """
    cb_kws_default = {'axloc': None, 'cbrange': None, 'n_levels': None, 'label_kws': {'size': 16},
                      'tick_params_kws': {'labelsize': 16}}

    # Load default kwargs
    for k, v in cb_kws_default.items():
        if k not in cb_kws:
            cb_kws[k] = v

    if cb_kws['label'] != '':
        cb_kws['label_kws'] = cb_kws.get('label_kws', {})
        cb_kws['label_kws']['label'] = cb_kws.pop('label')

    cb_kws['cmap'] = _set_cmap(cb_kws['cmap'], cb_kws['n_levels'])

    return cb_kws


def _cmap_discretize(cmap_in, N):
    """Return a discrete colormap from a continuous colormap.

    Parameters:
        cmap_in:
            `matplotlib.cm <http://matplotlib.org/api/cm_api.html>`_ (colormap)
            object.
        N (int):
            Number of colors.

    Returns:
        `matplotlib.cm <http://matplotlib.org/api/cm_api.html>`_ object

    Example:
        >>> fig, ax = plt.subplots()
        >>> im = np.resize(np.arange(100), (5, 100))
        >>> dRdBu = _cmap_discretize(cm.RdBu, 5)
        >>> ax.imshow(im, cmap=dRdBu)
    """
    try:
        return cmap_in._resample(N)
    except AttributeError:
        cdict = cmap_in._segmentdata.copy()
        # N colors
        colors_i = np.linspace(0, 1., N)
        # N+1 indices
        indices = np.linspace(0, 1., N + 1)
        for key in ('red', 'green', 'blue'):
            # Find the N colors
            D = np.array(cdict[key])
            I = interpolate.interp1d(D[:, 0], D[:, 1])
            colors = I(colors_i)
            # Place these colors at the correct indices.
            A = np.zeros((N + 1, 3), float)
            A[:, 0] = indices
            A[1:, 1] = colors
            A[:-1, 2] = colors
            # Create a tuple for the dictionary.
            L = []
            for l in A:
                L.append(tuple(l))
            cdict[key] = tuple(L)

        return LinearSegmentedColormap('colormap', cdict, 1024)


def _reverse_cmap(cdict):
    """Reverse colormap dictionary."""
    cdict_r = {}
    for k, v in cdict.items():
        out = []
        for it in v:
            out.append((1 - it[0], it[1], it[2]))
        cdict_r[k] = sorted(out)
    return cdict_r


def _linearlab_filename():
    """Get filename and path for linearlab colormap."""
    return join(os.path.dirname(marvin.__file__), 'data', 'linearlab.csv')


def linearlab():
    """Make linearlab color map.

    `Description of linearlab palatte
    <https://mycarta.wordpress.com/2012/12/06/the-rainbow-is-deadlong-live-the-rainbow-part-5-cie-lab-linear-l-rainbow/>`_.

    Returns:
        cm, cm_r (tuple):
        `matplotlib.cm <http://matplotlib.org/api/cm_api.html>`_ object and reversed
        `matplotlib.cm <http://matplotlib.org/api/cm_api.html>`_ object
    """
    linearlab_file = _linearlab_filename()
    LinL = np.loadtxt(linearlab_file, delimiter=',')

    b3 = LinL[:, 2]  # value of blue at sample n
    b2 = LinL[:, 2]  # value of blue at sample n
    b1 = np.linspace(0, 1, len(b2))  # position of sample n - ranges from 0 to 1

    # setting up columns for list
    g3 = LinL[:, 1]
    g2 = LinL[:, 1]
    g1 = np.linspace(0, 1, len(g2))

    r3 = LinL[:, 0]
    r2 = LinL[:, 0]
    r1 = np.linspace(0, 1, len(r2))

    # creating list
    R = zip(r1, r2, r3)
    G = zip(g1, g2, g3)
    B = zip(b1, b2, b3)

    # transposing list
    RGB = zip(R, G, B)
    rgb = zip(*RGB)

    # creating dictionary
    k = ['red', 'green', 'blue']
    LinearL = dict(zip(k, rgb))

    LinearL_r = _reverse_cmap(LinearL)

    cmap = LinearSegmentedColormap('linearlab', LinearL)
    cmap_r = LinearSegmentedColormap('linearlab_r', LinearL_r)

    return (cmap, cmap_r)


def _get_cmap_rgb(cmap, n_colors=256):
    """Return RGB values of a colormap.

    Parameters:
        cmap:
            `matplotlib.cm <http://matplotlib.org/api/cm_api.html>`_ (colormap)
            object
        n_colors (int):
            Number of color tuples in colormap. Default is ``256``.

    Returns:
        array
    """
    rgb = np.zeros((n_colors, 3))
    for i in range(n_colors):
        rgb[i] = cmap(i)[:3]
    return rgb


def _output_cmap_rgb(cmap, path=None, n_colors=256):
    """Print RGB values of a colormap to a file.

    Parameters:
        cmap:
            `matplotlib.cm <http://matplotlib.org/api/cm_api.html>`_ (colormap)
            object
        path (str):
            Path to generate output file. Default is ``None``.
        n_colors (int):
            Number of color tuples in colormap. Default is ``256``.
    """
    rgb = _get_cmap_rgb(cmap, n_colors)
    if path is None:
        home = os.path.expanduser('~')
        path = join(home, 'Downloads')
    filename = join(path, '{}.txt'.format(cmap.name))
    header = '{:22} {:24} {:22}'.format('Red', 'Green', 'Blue')
    np.savetxt(filename, rgb, header=header)
    print('Wrote: {}'.format(filename))


def _one_color_cmap(color):
    """Generate a colormap with only one color.

    Useful for imshow.

    Parameters:
        color (str):
            Color.

    Returns:
        `matplotlib.cm <http://matplotlib.org/api/cm_api.html>`_ (colormap)
        object
    """
    cmap, ig = from_levels_and_colors(levels=(0, 1), colors=(color,))
    return cmap
