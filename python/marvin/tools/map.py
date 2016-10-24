#!/usr/bin/env python
# encoding: utf-8
#
# map.py
#
# Created by José Sánchez-Gallego on 26 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from distutils import version
import warnings

from astropy.io import fits
import numpy

import marvin
import marvin.api.api
import marvin.core.marvin_pickle
import marvin.core.exceptions
import marvin.tools.maps

try:
    import matplotlib.pyplot as plt
    import mpl_toolkits.axes_grid1
    pyplot = True
except ImportError:
    pyplot = False


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
        self.value = numpy.array(value).reshape(self.shape)

        if self.maps_property.ivar:
            fullname_ivar = self.maps_property.fullname(channel=self.channel, ext='ivar')
            ivar = mdb.session.query(getattr(table, fullname_ivar)).filter(
                table.file_pk == self.maps.data.pk).order_by(table.spaxel_index).all()
            self.ivar = numpy.array(ivar).reshape(self.shape)

        if self.maps_property.mask:
            fullname_mask = self.maps_property.fullname(channel=self.channel, ext='mask')
            mask = mdb.session.query(getattr(table, fullname_mask)).filter(
                table.file_pk == self.maps.data.pk).order_by(table.spaxel_index).all()
            self.mask = numpy.array(mask).reshape(self.shape)

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

        self.value = numpy.array(data['value'])
        self.ivar = numpy.array(data['ivar'])
        self.mask = numpy.array(data['mask'])
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
            path = os.path.join(path+'.mpf')

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

        if show_masked is False and array != 'mask':
            data = numpy.ma.array(data, mask=(self.mask > 0))

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
