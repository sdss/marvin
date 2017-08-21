# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-21 17:11:22
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-08-21 18:50:52

from __future__ import print_function, division, absolute_import
from marvin import config
from marvin.utils.dap import datamodel
from marvin.tools.query_utils import QueryParameter
from marvin.utils.dap.datamodel.base import Property
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np


def _compute_stats(data):
    ''' Compute some statistics '''
    stats = {'mean': np.nanmean(data), 'std': np.nanstd(data), 'median': np.nanmedian(data),
             'per10': np.nanpercentile(data, 10), 'per25': np.nanpercentile(data, 10),
             'per75': np.nanpercentile(data, 10), 'per90': np.nanpercentile(data, 10)}

    return stats


def _make_masked(data, mask=None):
    ''' Makes a masked array '''

    arr_data = data
    if not isinstance(data, np.ma.MaskedArray):
        # mask out NaN values if a mask not provided
        mask = mask if mask else np.isnan(data)
        # create array
        arr_data = np.ma.MaskedArray(data, mask=mask)

    return arr_data


def _create_figure(hist=None, hist_axes_visible=None):
    ''' Create a generic figure and axis '''
    # create the figure and axes
    if hist:
        fig = plt.figure()
    else:
        fig, ax_scat = plt.subplots()
    ax_hist_x = None
    ax_hist_y = None

    # create histogram axes
    if hist:
        if hist is True:
            gs = GridSpec(4, 4)
            ax_scat = fig.add_subplot(gs[1:4, 0:3])
            ax_hist_x = fig.add_subplot(gs[0, 0:3])
            ax_hist_y = fig.add_subplot(gs[1:4, 3])
        elif hist == 'x':
            gs = GridSpec(2, 1, height_ratios=[1, 2])
            ax_scat = fig.add_subplot(gs[1])
            ax_hist_x = fig.add_subplot(gs[0])
        elif hist == 'y':
            gs = GridSpec(1, 2, width_ratios=[2, 1])
            ax_scat = fig.add_subplot(gs[0])
            ax_hist_y = fig.add_subplot(gs[1])

    # turn off histogram axes
    if ax_hist_x:
        plt.setp(ax_hist_x.get_xticklabels(), visible=hist_axes_visible)
    if ax_hist_y:
        plt.setp(ax_hist_y.get_yticklabels(), visible=hist_axes_visible)

    return fig, ax_scat, ax_hist_x, ax_hist_y


def _create_hist_title(data):
    ''' create a title for the histogram '''
    stats = _compute_stats(data)
    hist_title = 'Stats: $\\mu={mean:.3f}, \\sigma={std:.3f}$'.format(**stats)
    return hist_title


def _get_dap_datamodel_property_label(quantity):
    ''' Format a DAP datamodel property string label '''
    return '{0} [{1}]'.format(quantity.to_string('latex'), quantity.unit.to_string('latex'))


def _get_axis_label(column, axis=''):
    ''' Create an axis label '''

    if isinstance(column, QueryParameter):
        if hasattr(column, 'property') and column.property:
            label = _get_dap_datamodel_property_label(column.property)
        else:
            label = column.display
    elif isinstance(column, Property):
        label = _get_dap_datamodel_property_label(column)
    else:
        label = '{0} axis'.format(axis).strip()

    return label


def plot(x, y, **kwargs):
    ''' Scatter plot '''

    assert all([x, y]), 'Must provide both an x and y column'
    assert isinstance(x, (list, np.ndarray)), 'x data must be a list or Numpy array '
    assert isinstance(y, (list, np.ndarray)), 'y data must be a list or Numpy array '

    # general keyword arguments
    use_datamodel = kwargs.pop('usemodel', None)
    xmask = kwargs.pop('xmask', None)
    ymask = kwargs.pop('ymask', None)
    x_col = kwargs.pop('x_col', None)
    y_col = kwargs.pop('y_col', None)

    # scatterplot keyword arguments
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    color = kwargs.get('color', 'r')
    size = kwargs.get('size', 20)
    marker = kwargs.get('marker', 'o')
    edgecolors = kwargs.get('edgecolors', 'black')

    # histogram keywords
    hist = kwargs.get('hist', True)
    bins = kwargs.get('bins', [50, 50])
    hist_axes_visible = kwargs.get('hist_axes_visible', False)
    xhist_label = kwargs.get('xhist_label', 'Counts')
    yhist_label = kwargs.get('yhist_label', 'Counts')
    xhist_title = kwargs.get('xhist_title', None)
    yhist_title = kwargs.get('yhist_title', None)
    hist_color = kwargs.get('hist_color', 'lightblue')

    # convert to numpy masked arrays
    x = _make_masked(x, mask=xmask)
    y = _make_masked(y, mask=ymask)

    # create figure and axes objects
    fig, ax_scat, ax_hist_x, ax_hist_y = _create_figure(hist=hist, hist_axes_visible=hist_axes_visible)

    # set limits
    if xlim is not None:
        assert len(xlim) == 2, 'x-range must be a list or tuple of 2'
        ax_scat.set_xlim(*xlim)

    if ylim is not None:
        assert len(ylim) == 2, 'y-range must be a list or tuple of 2'
        ax_scat.set_ylim(*ylim)

    # set display names
    xlabel = xlabel if xlabel else _get_axis_label(x_col, axis='x')
    ylabel = ylabel if ylabel else _get_axis_label(y_col, axis='y')
    ax_scat.set_xlabel(xlabel)
    ax_scat.set_ylabel(ylabel)

    # create the scatter plot
    ax_scat.scatter(x, y, c=color, s=size, marker=marker, edgecolors=edgecolors, **kwargs)

    # set axes object
    axes = [ax_scat]

    # create histogram dictionary
    if hist:
        hist_data = {}
        xbin, ybin = bins if isinstance(bins, list) else (bins, None) if hist == 'x' else (None, bins)

    # set x-histogram
    if ax_hist_x:
        # set label
        ax_hist_x.set_ylabel(yhist_label)
        # set title
        xhist_title = xhist_title if xhist_title else _create_hist_title(x)
        ax_hist_x.set_title(xhist_title)
        # add to axes
        axes.append(ax_hist_x)
        # create histogram
        xhist, xbins, xpatches = ax_hist_x.hist(x[~x.mask], bins=xbin, color=hist_color,
                                                edgecolor=edgecolors, range=xlim, **kwargs)
        hist_data['xhist'] = {'counts': xhist, 'binedges': xbins, 'binsize': xbin, 'bins': np.digitize(x, xbins)}

    # set y-histogram
    if ax_hist_y:
        # set label
        ax_hist_y.set_xlabel(xhist_label)
        # set title
        yhist_title = yhist_title if yhist_title else _create_hist_title(y)
        ax_hist_y.set_ylabel(yhist_title)
        ax_hist_y.yaxis.set_label_position('right')
        ax_hist_y.yaxis.label.set_fontsize(12.0)
        # add to axes
        axes.append(ax_hist_y)
        # create histogram
        yhist, ybins, ypatches = ax_hist_y.hist(y[~y.mask], bins=ybin, orientation='horizontal',
                                                color=hist_color, edgecolor=edgecolors, range=ylim, **kwargs)
        hist_data['yhist'] = {'counts': yhist, 'binedges': ybins, 'binsize': ybin, 'bins': np.digitize(y, ybins)}

    output = (fig, axes, hist_data) if hist else (fig, axes)
    return output

