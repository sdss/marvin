# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-21 17:11:22
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-29 15:49:24

from __future__ import print_function, division, absolute_import
from marvin import config
from marvin.utils.datamodel.dap import datamodel
from marvin.core.exceptions import MarvinUserWarning
from marvin.utils.datamodel.query.base import QueryParameter
from marvin.utils.datamodel.dap.base import Property
from marvin.utils.general import invalidArgs, isCallableWithArgs
from matplotlib.gridspec import GridSpec
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import six


def compute_stats(data):
    ''' Compute some statistics given a data array

    Computes some basic statistics given a data array, excluding NaN values.
    Computes and returns the following Numpy statistics: mean, standard deviation,
    median, and the 10th, 25th, 75th, and 90th percentiles.

    Parameters:
        data (list|ndarray):
            A list or Numpy array of data

    Returns:
        A dictionary of statistics values

    '''
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
    stats = compute_stats(data)
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
    elif isinstance(column, six.string_types):
        label = column
    else:
        # label = '{0} axis'.format(axis).strip()
        label = ''

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
    if isCallableWithArgs(func, new_kwargs):
        return new_kwargs
    else:
        raise MarvinUserWarning('Cannot call func {0} with current kwargs {1}. Check your inputs'.format(func, new_kwargs))


def plot(x, y, **kwargs):
    ''' Create a scatter plot given two columns of data

    Creates a Matplotlib scatter plot using two input arrays of data.  By default, will also
    create and dispay histograms for the x and y data.  This can be disabled setting the "with_hist"
    keyword to False, or "x", or "y" for displaying only that column. Accepts all the same keyword
    arguments as matplotlib scatter and hist methods.

    Parameters:
        x (list|ndarray):
            The x array of data
        y (list|ndarray):
            The y array of data
        xmask (ndarray):
            A mask to apply to the x-array of data
        ymask (ndarray):
            A mask to apply to the y-array of data
        with_hist (bool|str):
            If True, creates the plot with both x,y histograms.  False, disables it.  If 'x' or 'y',
            only creates that histogram.  Default is True.
        hist_axes_visible (bool):
            If True, disables the x-axis ticks for each histogram.  Default is True.
        xlim (tuple):
            A tuple limited the range of the x-axis
        ylim (tuple):
            A tuple limited the range of the y-axis
        xlabel (str|Marvin column):
            The x axis label or a Marvin DataModel Property or QueryParameter to use for display
        ylabel (str|Marvin column):
            The y axis label or a Marvin DataModel Property or QueryParameter to use for display
        bins (int|tuple):
            A number or tuple specifying the number of bins to use in the histogram.  Default is 50.  An integer
            number is adopted for both x and y bins.  A tuple is used to customize per axis.
        kwargs (dict):
            Any other keyword arguments to be passed to `matplotlib.pyplot.scatter
                <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter>`_ or
            `matplotlib.pyplot.hist<http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist>`.

    Returns:
        A tuple of the matplotlib figure, axes, and histogram data (if returned)

    Example:
        >>> # create a scatter plot
        >>> import numpy as np
        >>> from marvin.utils.scatter import plot
        >>> x = np.random.random(100)
        >>> y = np.random.random(100)
        >>> plot(x, y)
    '''

    assert np.all([x, y]), 'Must provide both an x and y column'
    assert isinstance(x, (list, np.ndarray)), 'x data must be a list or Numpy array '
    assert isinstance(y, (list, np.ndarray)), 'y data must be a list or Numpy array '

    # general keyword arguments
    use_datamodel = kwargs.pop('usemodel', None)
    xmask = kwargs.pop('xmask', None)
    ymask = kwargs.pop('ymask', None)

    # scatterplot keyword arguments
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    color = kwargs.pop('color', 'r')
    size = kwargs.pop('size', 20)
    marker = kwargs.pop('marker', 'o')
    edgecolors = kwargs.pop('edgecolors', 'black')

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
    xlabel = _get_axis_label(xlabel, axis='x')
    ylabel = _get_axis_label(ylabel, axis='y')
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
        xbin, ybin = bins if isinstance(bins, list) else (bins, bins)

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
    ''' Create a histogram of an array

    Plots a histogram of an input column of data.  Input can be a list or a Numpy
    array.  Converts the input into a Numpy MaskedArray, applying the optional mask.  If no
    mask is supplied, it masks any NaN values.  Accepts all the same keyword arguments as
    matplotlib hist method.

    Also computes and returns a dictionary of histogram data.  The dictionary includes the following
    keys:
        bins - The number of bins used
        counts - A list of the count of objects within each bin
        binedges - A list of the left binedge used in defining each bin
        binids - An array of the same shape as input data, containing the binid of each element
        indices - A dictionary of a list of array indices within each bin

    Parameters:
        data (list|ndarray):
            A column of data to plot with.  Required.
        mask (ndarray):
            A mask to use on the data, applied to the data in a Numpy Masked Array.
        fig (plt.fig):
            An optional matplotlib figure object
        ax (plt.ax):
            An optional matplotlib axis object
        bins (int):
            The number of bins to use.  Default is 50 bins.
        xlabel (str|Marvin Column):
            The x axis label or a Marvin DataModel Property or QueryParameter to use for display
        ylabel (str):
            The y axis label
        title (str):
            The plot title
        rotate_title (bool):
            If True, moves the title text to the right y-axis during a horizontal histogram.  Default is False.
        return_fig (bool):
            If True, return the figure and axis object.  Default is True.
        kwargs (dict):
            Any other keyword arguments to be passed to `matplotlib.pyplot.hist
                <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist>`_.

    Returns:
        Tuple of histogram data, the matplotlib figure and axis objects

    Example:
        >>> # histogram some random data
        >>> from marvin.utils.plot.scatter import hist
        >>> import numpy as np
        >>> x = np.random.random(100)
        >>> hist_data, fig, ax = hist(x)
    '''

    assert isinstance(data, (list, np.ndarray)), 'data must be a list or Numpy array '
    data = _make_masked(data, mask=mask)

    # general keywords
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
    xlabel = _get_axis_label(xlabel, axis='x')
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
        ax.yaxis.set_label_position('right')
        ax.yaxis.label.set_fontsize(12.0)
        ax.set_ylabel(title, rotation=270, verticalalignment='bottom')

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

    hist_data = {'counts': counts, 'binedges': binedges, 'bins': bins,
                 'binids': binids, 'indices': indices}

    output = (hist_data, fig, ax) if return_fig else hist_data
    return output



