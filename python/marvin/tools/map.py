#!/usr/bin/env python
# encoding: utf-8
#
# map.py
#
# Created by José Sánchez-Gallego on 26 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re

import numpy

import marvin
import marvin.api.api
import marvin.core.exceptions
import marvin.tools.maps

try:
    import matplotlib.pyplot as plt
    import mpl_toolkits.axes_grid1
    pyplot = True
except:
    pyplot = False

try:
    import sqlalchemy
except:
    sqlalchemy = None


class Map(object):
    """Describes a single DAP map in a Maps object.

    Unlike a ``Maps`` object, which contains all the information from a DAP
    maps file, this class represents only one of the multiple 2D maps contained
    within. For instance, ``Maps`` may contain emission line maps for multiple
    channels. A ``Map`` would be, for example, the map for ``EMLINE_GFLUX`` and
    channel ``Ha-6564``.

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
        category (str):
            The category of the map to be extractred. E.g., `'EMLINE_GFLUX'`.
        channel (str or None):
            If the ``category`` contains multiple channels, the channel to use,
            e.g., ``Ha-6564'. Otherwise, ``None``.

    """

    # TODO: make this a MarvinToolsClass (JSG)

    def __init__(self, maps, category, channel=None):

        assert isinstance(maps, marvin.tools.maps.Maps)

        self.parent_maps = maps
        self.category = category.upper()
        self.channel = channel.upper() if channel is not None else None

        self.value = None
        self.header = None
        self.unit = None
        self.index = None

        if maps.data_origin == 'file':
            self._init_from_file()
        elif maps.data_origin == 'db':
            self._init_from_db()
        elif maps.data_origin == 'api':
            self._init_from_remote()

        # In some cases ivar and mask are None. We make them arrays.
        if self.ivar is None:
            self.ivar = numpy.zeros(self.value.shape)

        if self.mask is None:
            self.mask = numpy.ones(self.value.shape)

    def _init_from_file(self):
        """Initialises de Map from a ``Maps`` with ``data_origin='file'``."""

        if self.category not in self.parent_maps.data:
            raise marvin.core.exceptions.MarvinError(
                'invalid category {0}'.format(self.category))

        self.header = self.parent_maps.data[self.category].header
        self.unit = self.header['BUNIT'] if 'BUNIT' in self.header else None

        # Gets the channels and creates the names.
        channel_keys = [key for key in self.header.keys() if re.match('C[0-9]+', key)]
        names = [re.sub('\-+', '-', self.header[key]) for key in channel_keys]
        names_upper = [name.upper() for name in names]

        if len(names_upper) > 0 and self.channel is None:
            raise marvin.core.exceptions.MarvinError(
                'a channel is required to initialise a map for category {0}'
                .format(self.category))

        if self.channel not in names_upper:
            raise marvin.core.exceptions.MarvinError(
                'channel {0} not found for category {1}'.format(self.channel,
                                                                self.category))

        self.index = names_upper.index(self.channel)

        self.value = self.parent_maps.data[self.category].data[self.index, :, :]
        self.ivar = self.parent_maps.data[self.category + '_ivar'].data[self.index, :, :]
        self.mask = self.parent_maps.data[self.category + '_mask'].data[self.index, :, :]

        return

    def _init_from_db(self):
        """Initialises de Map from a ``Maps`` with ``data_origin='db'``."""

        # TODO: this will break if the datamodel changes ... (JSG)

        assert sqlalchemy, 'sqlalchemy is required.'

        mdb = marvin.marvindb

        dap_db_file = self.parent_maps.data

        # Depending on the type of category we'll need to run a different query.
        if 'EMLINE' in self.category:
            assert self.channel is not None, 'channel required for {0}'.format(self.category)
            subcategory = self.category.split('_')[1]
            emline_name, emline_wavelength = self.channel.split('-')

            emline = mdb.session.query(mdb.dapdb.EmLine).join(
                mdb.dapdb.File,
                mdb.dapdb.EmLineType,
                mdb.dapdb.EmLineParameter).filter(
                    mdb.dapdb.File.pk == dap_db_file.pk,
                    sqlalchemy.func.upper(mdb.dapdb.EmLineType.name) == emline_name,
                    mdb.dapdb.EmLineType.rest_wavelength == float(emline_wavelength),
                    mdb.dapdb.EmLineParameter.name == subcategory).first()

            if emline is None:
                raise marvin.core.exceptions.MarvinError('no results found')

            self.value = numpy.array(emline.value)
            self.ivar = numpy.array(emline.ivar)
            self.mask = numpy.array(emline.mask)
            self.unit = emline.parameter.unit

        elif 'STELLAR' in self.category:
            subcategory = self.category.split('_')[1]
            assert subcategory in ['VEL', 'SIGMA']

            stellar = mdb.session.query(mdb.dapdb.StellarKin).join(
                mdb.dapdb.File, mdb.dapdb.StellarKinParameter).filter(
                    mdb.dapdb.File.pk == dap_db_file.pk,
                    mdb.dapdb.StellarKinParameter.name == subcategory).first()

            if stellar is None:
                raise marvin.core.exceptions.MarvinError('no results found')

            self.value = numpy.array(stellar.value)
            self.ivar = numpy.array(stellar.ivar)
            self.mask = numpy.array(stellar.mask)
            self.unit = stellar.parameter.unit

        elif 'SPECINDEX' in self.category:
            assert self.channel is not None, 'channel required for {0}'.format(self.category)

            specindex = mdb.session.query(mdb.dapdb.SpecIndex).join(
                mdb.dapdb.File,
                mdb.dapdb.SpecIndexType).filter(
                    mdb.dapdb.File.pk == dap_db_file.pk,
                    sqlalchemy.func.upper(mdb.dapdb.SpecIndexType.name) == self.channel).first()

            if specindex is None:
                raise marvin.core.exceptions.MarvinError('no results found')

            self.value = numpy.array(specindex.value)
            self.ivar = numpy.array(specindex.ivar)
            self.mask = numpy.array(specindex.mask)
            self.unit = None

        else:
            marvin.core.exceptions.MarvinError(
                'category {0} is not valid or I do not know how to parse it.'
                .format(self.category))

        return

    def _init_from_remote(self):
        """Initialises de Map from a ``Maps`` with ``data_origin='api'``."""

        url = marvin.config.urlmap['api']['getmap']['url']

        url_full = url.format(
            **{'name': self.parent_maps.plateifu,
               'path': 'category={0}/channel={1}'.format(self.category,
                                                         self.channel)})

        try:
            response = marvin.api.api.Interaction(url_full)
        except Exception as ee:
            raise marvin.core.exceptions.MarvinError(
                'found a problem when getting the map: {0}'.format(str(ee)))

        data = response.getData()

        if data is None:
            raise marvin.core.exceptions.MarvinError(
                'something went wrong. '
                'Error is: {0}'.format(response.results['error']))

        self.value = numpy.array(data['value'])
        self.ivar = numpy.array(data['ivar'])
        self.mask = numpy.array(data['mask'])
        self.unit = data['unit']

        return

    def plot(self, array='value', xlim=None, ylim=None, zlim=None,
             xlabel=None, ylabel=None, zlabel=None, cmap=None, kw_imshow=None,
             figure=None, return_figure=False):
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
                variance or the mask.
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

        Returns:
            ax (`matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_):
                The matplotlib axes object containing the plot representing
                the map. If ``return_figure=True``, a tuple will be
                returned of the form ``(ax, fig)``.

        Example:

          >>> maps = Maps(plateifu='8485-1901')
          >>> ha_map = maps.getMap(category='emline_gflux', channel='ha-6564')
          >>> ha_map.plot()

        .. |axes| replace:: matplotlib.axes
        .. _axes: http://matplotlib.org/api/axes_api.html

        """

        # TODO: plot in sky coordinates. (JSG)

        if not pyplot:
            raise marvin.core.exceptions.MarvinMissingDependency(
                'matplotlib is not installed.')

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
