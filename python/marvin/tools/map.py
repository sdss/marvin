#!/usr/bin/env python
# encoding: utf-8
#
# Licensed under a 3-clause BSD license.
#
# map.py
#
# Created by José Sánchez-Gallego on 26 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from distutils import version
import os
import warnings

from astropy.io import fits
from astropy.units import Quantity

import numpy as np

import marvin
import marvin.api.api
import marvin.core.marvin_pickle
import marvin.core.exceptions
import marvin.tools.maps
import marvin.utils.plot.map

from marvin.utils.dap.datamodel.base import Property

try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None


class Map(Quantity):
    """Describes a single DAP map in a Maps object.

    Unlike a ``Maps`` object, which contains all the information from a DAP
    maps file, this class represents only one of the multiple 2D maps contained
    within. For instance, ``Maps`` may contain emission line maps for multiple
    channels. A ``Map`` would be, for example, the map for ``emline_gflux`` and
    channel ``ha_6564``.

    A ``Map`` returns and astropy 2D Quantity-like array with additional
    attributes for ``ivar`` and ``mask``.

    ``Map`` objects are not intended to be initialised directly, at least
    for now. To get a ``Map`` instance, use the
    :func:`~marvin.tools.maps.Maps.getMap` method.

    Parameters:
        maps (:class:`~marvin.tools.maps.Maps` object):
            The :class:`~marvin.tools.maps.Maps` instance from which we
            are extracting the ``Map``.
        property_name (str):
            The category of the map to be extractred (e.g., ``'emline_gflux'``)
            or the full property name including channel
            (``'emline_gflux_ha_6564'``).
        channel (str or None):
            If the ``property`` contains multiple channels, the channel to use,
            e.g., ``ha_6564'. Otherwise, ``None``.

    """

    def __new__(cls, maps, property_name, channel=None, dtype=None, copy=True):

        assert isinstance(maps, marvin.tools.maps.Maps)

        maps = maps
        datamodel = maps._datamodel

        if isinstance(property_name, Property):
            prop = property_name
        else:
            prop = maps._match_properties(property_name, channel=channel)

        assert prop in datamodel.properties, \
            'failed sanity check. Property does not match.'

        channel = prop.channel

        release = maps.release

        if maps.data_origin == 'file':
            value, ivar, mask, header = cls._get_from_file(maps, prop)
        elif maps.data_origin == 'db':
            value, ivar, mask, header = cls._get_from_db(maps, prop)
        elif maps.data_origin == 'api':
            value, ivar, mask, header = cls._get_from_api(maps, prop)

        unit = prop.unit
        value = value * prop.scale

        obj = Quantity(value, unit=unit, dtype=dtype, copy=copy)
        obj = obj.view(cls)
        obj._set_unit(unit)

        obj.property = prop
        obj.channel = channel

        obj.header = header
        obj.scale = 1

        obj.maps = maps
        obj.release = release
        obj._datamodel = datamodel

        obj.ivar = (np.array(ivar) / (prop.scale ** 2)) if ivar is not None else None
        obj.mask = np.array(mask) if mask is not None else None

        return obj

    def __getitem__(self, sl):

        new_obj = super(Map, self).__getitem__(sl)

        if type(new_obj) is not type(self):
            new_obj = self._new_view(new_obj)

        new_obj._set_unit(self.unit)

        new_obj.ivar = self.ivar.__getitem__(sl) if self.ivar is not None else self.ivar
        new_obj.mask = self.mask.__getitem__(sl) if self.mask is not None else self.mask

        return new_obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.property = getattr(obj, 'property', None)
        self.channel = getattr(obj, 'channel', None)
        self.maps = getattr(obj, 'maps', None)
        self.release = getattr(obj, 'release', None)
        self._datamodel = getattr(obj, '_datamodel', None)

        self.ivar = getattr(obj, 'ivar', None)
        self.mask = getattr(obj, 'mask', None)

    @property
    def error(self):
        """The standard deviation of the measurement."""

        if self.ivar is None:
            return None

        np.seterr(divide='ignore')

        return np.sqrt(1. / self.ivar) * self.unit

    @property
    def masked(self):
        """Returns a masked array."""

        assert self.mask is not None, 'mask is None'

        return np.ma.array(self.value, mask=self.mask > 0)

    def __repr__(self):

        if np.isscalar(self.value):
            return super(Map, self).__repr__()
        else:
            return ('<Marvin Map (plateifu={0.maps.plateifu!r}, property={1!r}, '
                    'channel={0.channel!r})>\n{2!r} {3}'.format(self, self.property.name,
                                                                self.value, self.unit.to_string()))

    @property
    def snr(self):
        """Returns the signal-to-noise ratio for each spaxel in the map."""

        return np.abs(self.value * np.sqrt(self.ivar))

    @staticmethod
    def _get_from_file(maps, prop):
        """Initialises the Map from a ``Maps`` with ``data_origin='file'``."""

        header = maps.data[prop.name].header

        if prop.idx is not None:
            channel_idx = prop.idx
            value = maps.data[prop.name].data[channel_idx]
            if prop.ivar:
                ivar = maps.data[prop.name + '_ivar'].data[channel_idx]
            if prop.mask:
                mask = maps.data[prop.name + '_mask'].data[channel_idx]
        else:
            value = maps.data[prop.name].data
            if prop.ivar:
                ivar = maps.data[prop.name + '_ivar'].data
            if prop.mask:
                mask = maps.data[prop.name + '_mask'].data

        return value, ivar, mask, header

    @staticmethod
    def _get_from_db(maps, prop):
        """Initialises the Map from a ``Maps`` with ``data_origin='db'``."""

        mdb = marvin.marvindb

        if not mdb.isdbconnected:
            raise marvin.core.exceptions.MarvinError('No db connected')

        if sqlalchemy is None:
            raise marvin.core.exceptions.MarvinError('sqlalchemy required to access the local DB.')

        if version.StrictVersion(maps._dapver) <= version.StrictVersion('1.1.1'):
            table = mdb.dapdb.SpaxelProp
        else:
            table = mdb.dapdb.SpaxelProp5

        fullname_value = prop.db_column()
        value = mdb.session.query(getattr(table, fullname_value)).filter(
            table.file_pk == maps.data.pk).order_by(table.spaxel_index).all()
        value = np.array(value).reshape(maps.shape).T

        if prop.ivar:
            fullname_ivar = prop.db_column(ext='ivar')
            ivar = mdb.session.query(getattr(table, fullname_ivar)).filter(
                table.file_pk == maps.data.pk).order_by(table.spaxel_index).all()
            ivar = np.array(ivar).reshape(maps.shape).T

        if prop.mask:
            fullname_mask = prop.db_column(ext='mask')
            mask = mdb.session.query(getattr(table, fullname_mask)).filter(
                table.file_pk == maps.data.pk).order_by(table.spaxel_index).all()
            mask = np.array(mask).reshape(maps.shape).T

        # Gets the header
        hdus = maps.data.hdus
        header_dict = None
        for hdu in hdus:
            if prop.name.upper() == hdu.extname.name.upper():
                header_dict = hdu.header_to_dict()
                break

        if not header_dict:
            warnings.warn('cannot find the header for property {0}.'
                          .format(prop.name),
                          marvin.core.exceptions.MarvinUserWarning)
        else:
            header = fits.Header(header_dict)

        return value, ivar, mask, header

    @staticmethod
    def _get_from_api(maps, prop):
        """Initialises the Map from a ``Maps`` with ``data_origin='api'``."""

        url = marvin.config.urlmap['api']['getmap']['url']

        url_full = url.format(
            **{'name': maps.plateifu,
               'property_name': prop.name,
               'channel': prop.channel,
               'bintype': maps.bintype.name,
               'template': maps.template.name})

        try:
            response = marvin.api.api.Interaction(url_full,
                                                  params={'release': maps._release})
        except Exception as ee:
            raise marvin.core.exceptions.MarvinError(
                'found a problem when getting the map: {0}'.format(str(ee)))

        data = response.getData()

        if data is None:
            raise marvin.core.exceptions.MarvinError(
                'something went wrong. Error is: {0}'.format(response.results['error']))

        value = np.array(data['value'])
        ivar = np.array(data['ivar']) if data['ivar'] is not None else None
        mask = np.array(data['mask']) if data['mask'] is not None else None
        header = fits.Header(data['header'])

        return value, ivar, mask, header

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
        unplickled. Note that, for map objects instantiated from a Maps object
        with ``data_origin='file'``, the original file must exists and be
        in the same path as when the object was first created.

        """

        return marvin.core.marvin_pickle.restore(path, delete=delete)

    def plot(self, *args, **kwargs):
        """Convenience method that wraps :func:`marvin.utils.plot.map.plot`.

        See :func:`marvin.utils.plot.map.plot` for full documentation.
        """
        return marvin.utils.plot.map.plot(dapmap=self, *args, **kwargs)
