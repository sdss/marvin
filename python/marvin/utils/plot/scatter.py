# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-21 17:11:22
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-08-22 11:47:24

from __future__ import print_function, division, absolute_import
from marvin import config
from marvin.utils.dap import datamodel
from marvin.core.exceptions import MarvinUserWarning
from marvin.tools.query_utils import QueryParameter
from marvin.utils.dap.datamodel.base import Property
from marvin.utils.general import invalidArgs, isCallableWithArgs
from matplotlib.gridspec import GridSpec
from collections import defaultdict, OrderedDict
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


def _format_hist_kwargs(axis, **kwargs):
    ''' Format the histogram kwargs from plot '''
    kwargs['color'] = kwargs.get('hist_color', 'lightblue')
    if axis == 'x':
        kwargs['ylabel'] = kwargs.get('xhist_label', 'Counts')
        kwargs['title'] = kwargs.get('xhist_title', None)
    elif axis == 'y':
        kwargs['ylabel'] = kwargs.get('yhist_label', 'Counts')
        kwargs['title'] = kwargs.get('yhist_title', None)
    kwargs['color'] = kwargs.get('hist_color', 'lightblue')
    kwargs['edgecolor'] = kwargs.get('edgecolors', 'black')
    return kwargs


def _prep_func_kwargs(func, kwargs):
    ''' Prepare the keyword arguments for the proper function input '''
    invalid = invalidArgs(func, kwargs)
    new_kwargs = kwargs.copy()
    for key in invalid:
        __ = new_kwargs.pop(key)
    print('new', new_kwargs, func, isCallableWithArgs(func, new_kwargs))
    if isCallableWithArgs(func, new_kwargs):
        return new_kwargs
    else:
        raise MarvinUserWarning('Cannot call func {0} with current kwargs {1}. Check your inputs'.format(func, new_kwargs))


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
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    color = kwargs.pop('color', 'r')
    size = kwargs.pop('size', 20)
    marker = kwargs.pop('marker', 'o')
    edgecolors = kwargs.get('edgecolors', 'black')

    # histogram keywords
    with_hist = kwargs.pop('with_hist', True)
    bins = kwargs.pop('bins', [50, 50])
    hist_axes_visible = kwargs.pop('hist_axes_visible', False)

    # convert to numpy masked arrays
    x = _make_masked(x, mask=xmask)
    y = _make_masked(y, mask=ymask)

    # create figure and axes objects
    fig, ax_scat, ax_hist_x, ax_hist_y = _create_figure(hist=with_hist, hist_axes_visible=hist_axes_visible)

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
    scat_kwargs = _prep_func_kwargs(plt.scatter, kwargs)
    ax_scat.scatter(x, y, c=color, s=size, marker=marker, edgecolors=edgecolors, **scat_kwargs)

    # set axes object
    axes = [ax_scat]

    # create histogram dictionary
    if with_hist:
        hist_data = {}
        xbin, ybin = bins if isinstance(bins, list) else (bins, None) if with_hist == 'x' else (None, bins)

    # set x-histogram
    if ax_hist_x:
        xhist_kwargs = _format_hist_kwargs('x', **kwargs)
        xhist, fig, ax_hist_x = hist(x, bins=xbin, fig=fig, ax=ax_hist_x, **xhist_kwargs)
        axes.append(ax_hist_x)
        hist_data['xhist'] = xhist

    # set y-histogram
    if ax_hist_y:
        yhist_kwargs = _format_hist_kwargs('y', **kwargs)
        yhist, fig, ax_hist_y = hist(y, bins=ybin, fig=fig, ax=ax_hist_y, orientation='horizontal',
                                     rotate_title=True, **yhist_kwargs)
        axes.append(ax_hist_y)
        hist_data['yhist'] = yhist

    output = (fig, axes, hist_data) if with_hist else (fig, axes)
    return output


def hist(data, mask=None, fig=None, ax=None, bins=None, **kwargs):
    ''' Create a histogram of an array '''

    assert isinstance(data, (list, np.ndarray)), 'data must be a list or Numpy array '
    data = _make_masked(data, mask=mask)

    # general keywords
    column = kwargs.pop('column', None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', 'Counts')
    title = kwargs.pop('title', None)
    rotate_title = kwargs.pop('rotate_title', False)
    return_fig = kwargs.pop('return_fig', True)

    # histogram keywords
    bins = bins if bins else 50
    color = kwargs.pop('color', 'lightblue')
    edgecolor = kwargs.pop('edgecolor', 'black')
    hrange = kwargs.pop('range', None)
    orientation = kwargs.pop('orientation', 'vertical')

    # create a figure and axis if they don't exist
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = plt.figure()

    # set labels
    xlabel = xlabel if xlabel else _get_axis_label(column, axis='x') if column else ''
    ax.set_ylabel(ylabel) if orientation == 'vertical' else ax.set_ylabel(xlabel)
    ax.set_xlabel(xlabel) if orientation == 'vertical' else ax.set_xlabel(ylabel)

    # reset the label positions
    ax.yaxis.set_label_position('left')
    ax.xaxis.set_label_position('bottom')

    # set title
    title = title if title else _create_hist_title(data)
    ax.set_title(title)

    if rotate_title:
        ax.set_title('')
        ax.set_ylabel(title)
        ax.yaxis.set_label_position('right')
        ax.yaxis.label.set_fontsize(12.0)

    # create histogram
    hist_kwargs = _prep_func_kwargs(plt.hist, kwargs)
    counts, binedges, patches = ax.hist(data[~data.mask], bins=bins, color=color,
                                        orientation=orientation, edgecolor=edgecolor,
                                        range=hrange, **hist_kwargs)

    # compute a dictionary of the binids containing a list of the array indices in each bin
    binids = np.digitize(data, binedges)
    dd = defaultdict(list)
    for i, binid in enumerate(binids):
        dd[binid].append(i)
    indices = OrderedDict(dd)

    hist_data = {'counts': counts, 'binedges': binedges, 'binsize': bins,
                 'binids': binids, 'indices': indices}

    output = (hist_data, fig, ax) if return_fig else hist_data
    return output



