# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-21 17:11:22
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last Modified time: 2018-11-08 16:21:30

from __future__ import absolute_import, division, print_function

import warnings
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import six
from astropy.visualization import hist as ahist
from matplotlib.gridspec import GridSpec

from marvin.core.exceptions import MarvinUserWarning
from marvin.utils.general import invalidArgs, isCallableWithArgs


try:
    import mpl_scatter_density as msd
except ImportError as e:
    msd = None
    msderr = ('mpl-scatter-density is required to plot large results and was not found.  '
              'To use this feature, please install the python package!')


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
             'per10': np.nanpercentile(data, 10), 'per25': np.nanpercentile(data, 25),
             'per75': np.nanpercentile(data, 75), 'per90': np.nanpercentile(data, 90)}

    return stats


def _make_masked(data, mask=None):
    ''' Makes a masked array '''

    arr_data = data
    if not isinstance(data, np.ma.MaskedArray):
        # mask out NaN values if a mask not provided
        warnings.warn("Masking out NaN values!", MarvinUserWarning)
        mask = mask if mask else np.isnan(data)
        # create array
        arr_data = np.ma.MaskedArray(data, mask=mask)

    return arr_data


def _create_figure(hist=None, hist_axes_visible=None, use_density=None):
    ''' Create a generic figure and axis '''

    # use a scatter density projection or not
    projection = 'scatter_density' if use_density else None

    # check if mpl-scatter-density if installed
    if not msd:
        raise ImportError(msderr)

    # create the figure
    fig = plt.figure()
    ax_hist_x = None
    ax_hist_y = None

    # create axes with or without histogram
    if hist:
        if hist is True:
            gs = GridSpec(4, 4)
            ax_scat = fig.add_subplot(gs[1:4, 0:3], projection=projection)
            ax_hist_x = fig.add_subplot(gs[0, 0:3])
            ax_hist_y = fig.add_subplot(gs[1:4, 3])
        elif hist == 'x':
            gs = GridSpec(2, 1, height_ratios=[1, 2])
            ax_scat = fig.add_subplot(gs[1], projection=projection)
            ax_hist_x = fig.add_subplot(gs[0])
        elif hist == 'y':
            gs = GridSpec(1, 2, width_ratios=[2, 1])
            ax_scat = fig.add_subplot(gs[0], projection=projection)
            ax_hist_y = fig.add_subplot(gs[1])
    else:
        ax_scat = fig.add_subplot(1, 1, 1, projection=projection)

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

    from marvin.utils.datamodel.query.base import QueryParameter
    from marvin.utils.datamodel.dap.base import Property

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


def _set_options():
    ''' Set some default Matplotlib options '''
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['grid.color'] = 'gray'
    mpl.rcParams['grid.linestyle'] = 'dashed'
    mpl.rcParams['grid.alpha'] = 0.8


def _set_limits(column, lim=None, sigma_cutoff=50, percent_clip=1):
    ''' Set an axis limit

    Determines whether to apply percentile clipping or not if any data
    has a zscore value above the sigma_cutoff value.  Applies percentile clipping
    centered around the mean.

    Parameters:
        column:
            The array of data to get limits of
        lim (list|tuple):
            A user provided range
        sigma_cutoff (int):
            The number of sigma away from the mean to cutoff
        percent_clip (int|tuple):
            The percent to clip off the data array.  Input values are taken as percentages.
            Can either be integer value (halved for lo,hi) or a tuple specifying lo,hi values.
            Default is 1%.

    Returns:
        A list of axis range values to use

    '''
    if lim is not None:
        assert len(lim) == 2, 'range must be a list or tuple of 2'
    else:
        # get percent clips
        if isinstance(percent_clip, (list, tuple)):
            lo, hi = percent_clip
        else:
            lo = percent_clip / 2.
            hi = 100 - lo

        zscore = stats.zscore(column)
        # use percentile limits if the max zscore is > 50 sigma away from mean/stdev
        if np.max(zscore) > sigma_cutoff:
            lim = [np.percentile(column, lo), np.percentile(column, hi)]
        else:
            pass
    return lim


def _check_input_data(coldim, col, data=None):
    ''' Check the input data

    Parameters:
        coldim (str):
            Name of the dimension
        col (str|array):
            The list or array of values.  If data keyword is specified, col is a string name
        data (Pandas.DataFrame)
            A Pandas dataframe

    Returns:
        The column of data
    '''

    # check data
    assert col is not None, 'Must provide an {0} column'.format(coldim)

    if data is not None:
        assert isinstance(col, str), '{0} must be a string name if Dataframe provided'.format(coldim)
        assert isinstance(data, pd.core.frame.DataFrame), 'data must be Pandas dataframe'
        assert col in data.columns, '{0} must be a specified column name in Pandas dataframe'.format(coldim)
        col = data[col]
    else:
        assert isinstance(col, (list, np.ndarray, pd.core.series.Series)), '{0} data must be a list, Pandas Series, or Numpy array'.format(coldim)

    return col


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
    kwargs['edgecolor'] = kwargs.get('edgecolors', None)
    return kwargs


def _prep_func_kwargs(func, kwargs):
    ''' Prepare the keyword arguments for the proper function input

    Checks an input dictionary against allowed keyword arguments
    for a given function.  Returns only those usable in that function.

    Parameters:
        func:
            The name of the function to check keywords against
        kwargs (dict):
            A dictionary of keyword arguments to test

    Returns:
        A new dictionary of usable keyword arguments

    '''
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

    Creates a Matplotlib plot using two input arrays of data.  Creates either a Matplotlib scatter
    plot, hexbin plot, or scatter density plot depending on the size of the input data.
    For data with < 1000 values, creates a scatter plot.  For data with values between
    1000 and 500,000, creates a hexbin plot.  For data with > 500,000 values, creates
    a scatter density plot.

    By default, will also create and display histograms for the x and y data.  This can be disabled
    setting the "with_hist" keyword to False, or "x", or "y" for displaying only that column.
    Accepts all the same keyword arguments as matplotlib scatter, hexbin, and hist methods.

    See `scatter-density <https://github.com/astrofrog/mpl-scatter-density>`_
    See `matplotlib.pyplot.scatter <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter>`_
    See `matplotlib.pyplot.hexbin <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hexbin>`_

    Parameters:
        x (str|list|ndarray):
            The x array of data
        y (str|list|ndarray):
            The y array of data
        data (Pandas dataframe):
            Optional Pandas Dataframe.  x, y specify string column names in the dataframe
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
        return_figure (bool):
            If True, return the figure and axis object.  Default is True.
        kwargs (dict):
            Any other keyword arguments to be passed to `matplotlib.pyplot.scatter <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter>`_
            or `matplotlib.pyplot.hist <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist>`_ or
            `matplotlib.pyplot.hexbin <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hexbin>`_.

    Returns:
        A tuple of the matplotlib figure, axes, and histogram data (if returned)

    Example:
        >>> # create a scatter plot
        >>> import numpy as np
        >>> from marvin.utils.plot.scatter import plot
        >>> x = np.random.random(100)
        >>> y = np.random.random(100)
        >>> plot(x, y)
    '''

    # check the input data
    data = kwargs.pop('data', None)
    x = _check_input_data('x', x, data=data)
    y = _check_input_data('y', y, data=data)

    # general keyword arguments
    use_datamodel = kwargs.pop('usemodel', None)
    xmask = kwargs.pop('xmask', None)
    ymask = kwargs.pop('ymask', None)
    return_figure = kwargs.pop('return_figure', True)

    # scatterplot keyword arguments
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    color = kwargs.pop('color', None)
    size = kwargs.pop('size', 20)
    marker = kwargs.pop('marker', 'o')
    edgecolors = kwargs.pop('edgecolors', 'black')

    # hexbin keywords
    gridsize = kwargs.pop('gridsize', 50)

    # histogram keywords
    with_hist = kwargs.pop('with_hist', True)
    bins = kwargs.pop('bins', ['scott', 'scott'])
    hist_axes_visible = kwargs.pop('hist_axes_visible', False)

    # convert to numpy masked arrays
    x = _make_masked(x, mask=xmask)
    y = _make_masked(y, mask=ymask)
    count = len(x)
    use_density = True if count > 500000 else False

    # create figure and axes objects
    with plt.style.context('seaborn-darkgrid'):
        fig, ax_scat, ax_hist_x, ax_hist_y = _create_figure(hist=with_hist, use_density=use_density,
                                                            hist_axes_visible=hist_axes_visible)

    # create the hexbin or scatter plot
    kind = kwargs.get('kind', None)
    assert kind in ['hex', 'scatter', 'density', 'joint', None], 'plot kind must be either scatter, hex, density, or joint'
    if count > 1000 and count <= 500000:
        scat_kwargs = _prep_func_kwargs(plt.hexbin, kwargs)
        main = ax_scat.hexbin(x, y, gridsize=gridsize, mincnt=1, cmap='inferno', **scat_kwargs)
        cb = fig.colorbar(main, ax=ax_scat, label='Counts')
        #ax_scat.grid(color='gray', linestyle='dashed', alpha=0.8)
    elif count > 500000:
        # abort if mpl-scatter-density is not installed
        if not msd:
            raise ImportError(msderr)

        scat_kwargs = _prep_func_kwargs(plt.imshow, kwargs)
        main = ax_scat.scatter_density(x, y, cmap='inferno', **scat_kwargs)
        cb = fig.colorbar(main, ax=ax_scat, label='Number of points per pixel')
        ax_scat.grid(color='gray', linestyle='dashed', alpha=0.8)
    else:
        # create the scatter plot
        scat_kwargs = _prep_func_kwargs(plt.scatter, kwargs)
        main = ax_scat.scatter(x, y, c=color, s=size, marker=marker, edgecolors=edgecolors, **scat_kwargs)
        cb = None
        #ax_scat.grid(color='gray', linestyle='dashed', alpha=0.8)

    # set limits
    xlim = _set_limits(x, lim=xlim)
    ylim = _set_limits(y, lim=ylim)
    if xlim:
        ax_scat.set_xlim(xlim)
    if ylim:
        ax_scat.set_ylim(ylim)

    # set display names
    xlabel = _get_axis_label(xlabel, axis='x')
    ylabel = _get_axis_label(ylabel, axis='y')
    ax_scat.set_xlabel(xlabel)
    ax_scat.set_ylabel(ylabel)

    # set axes object
    axes = [ax_scat]

    # create histogram dictionary
    if with_hist:
        hist_data = {}
        xbin, ybin = bins if isinstance(bins, list) else (bins, bins)

    # set x-histogram
    if ax_hist_x:
        xhist_kwargs = _format_hist_kwargs('x', **kwargs)
        #xrange = ax_scat.get_xlim()
        xhist, fig, ax_hist_x = hist(x, bins=xbin, fig=fig, ax=ax_hist_x, **xhist_kwargs)
        axes.append(ax_hist_x)
        hist_data['xhist'] = xhist
        if cb is not None:
            ocb = fig.colorbar(main, ax=ax_hist_x)
            ocb.remove()

    # set y-histogram
    if ax_hist_y:
        yhist_kwargs = _format_hist_kwargs('y', **kwargs)
        yhist, fig, ax_hist_y = hist(y, bins=ybin, fig=fig, ax=ax_hist_y, orientation='horizontal',
                                     rotate_title=True, **yhist_kwargs)
        axes.append(ax_hist_y)
        hist_data['yhist'] = yhist

    if return_figure:
        output = (fig, axes, hist_data) if with_hist else (fig, axes)
    else:
        output = hist_data if with_hist else None

    return output


def hist(arr, mask=None, fig=None, ax=None, bins=None, **kwargs):
    ''' Create a histogram of an array

    Plots a histogram of an input column of data.  Input can be a list or a Numpy
    array.  Converts the input into a Numpy MaskedArray, applying the optional mask.  If no
    mask is supplied, it masks any NaN values.  This uses
    `Astropy's enhanced hist <http://docs.astropy.org/en/stable/api/astropy.visualization.hist.html#astropy.visualization.hist>`_
    function under the hood. Accepts all the same keyword arguments as matplotlib hist method.

    Parameters:
        arr (list|ndarray):
            An array of data to plot with.  Required.
        mask (ndarray):
            A mask to use on the data, applied to the data in a Numpy Masked Array.
        fig (plt.fig):
            An optional matplotlib figure object
        ax (plt.ax):
            An optional matplotlib axis object
        bins (int):
            The number of bins to use.  Default is a `scott <http://docs.astropy.org/en/stable/visualization/histogram.html>`_ binning scheme.
        xlabel (str|Marvin Column):
            The x axis label or a Marvin DataModel Property or QueryParameter to use for display
        ylabel (str):
            The y axis label
        title (str):
            The plot title
        rotate_title (bool):
            If True, moves the title text to the right y-axis during a horizontal histogram.  Default is False.
        return_figure (bool):
            If True, return the figure and axis object.  Default is True.
        kwargs (dict):
            Any other keyword arguments to be passed to `matplotlib.pyplot.hist <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist>`_.

    Returns:
        tuple: histogram data, matplotlib figure, and axis objects.

        The histogram data returned is a dictionary containing::

            {
                'bins': The number of bins used,
                'counts': A list of the count of objects within each bin,
                'binedges': A list of the left binedge used in defining each bin,
                'binids': An array of the same shape as input data, containing the binid of each element,
                'indices': A dictionary of a list of array indices within each bin
            }

    Example:
        >>> # histogram some random data
        >>> from marvin.utils.plot.scatter import hist
        >>> import numpy as np
        >>> x = np.random.random(100)
        >>> hist_data, fig, ax = hist(x)
    '''

    # check the input data
    data = kwargs.pop('data', None)
    arr = _check_input_data('column', arr, data=data)
    arr = _make_masked(arr, mask=mask)

    # general keywords
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', 'Counts')
    title = kwargs.pop('title', None)
    rotate_title = kwargs.pop('rotate_title', False)
    return_figure = kwargs.pop('return_figure', True)

    # histogram keywords
    bins = bins if bins else 'scott'
    color = kwargs.pop('color', None)
    edgecolor = kwargs.pop('edgecolor', None)
    hrange = kwargs.pop('range', None)
    orientation = kwargs.pop('orientation', 'vertical')

    # create a figure and axis if they don't exist
    with plt.style.context('seaborn-darkgrid'):
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

    # set limits
    hrange = _set_limits(arr, lim=hrange)

    # set title
    title = title if title else _create_hist_title(arr)
    ax.set_title(title)

    if rotate_title:
        ax.set_title('')
        ax.yaxis.set_label_position('right')
        ax.yaxis.label.set_fontsize(12.0)
        ax.set_ylabel(title, rotation=270, verticalalignment='bottom')

    # create histogram
    hist_kwargs = _prep_func_kwargs(ahist, kwargs)
    counts, binedges, patches = ahist(arr[~arr.mask], bins=bins, color=color,
                                      orientation=orientation, edgecolor=edgecolor,
                                      range=hrange, ax=ax, **hist_kwargs)

    # compute a dictionary of the binids containing a list of the array indices in each bin
    binids = np.digitize(arr, binedges)
    inds = np.where(binids)[0]
    indices = defaultdict(list)
    tmp = list(map(lambda i, x: indices[x].append(i), inds, binids))

    hist_data = {'counts': counts, 'binedges': binedges, 'bins': bins,
                 'binids': binids, 'indices': indices}

    output = (hist_data, fig, ax) if return_figure else hist_data
    return output
