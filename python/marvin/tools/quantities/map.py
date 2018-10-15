#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-10-30
# @Filename: map.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by:   andrews
# @Last modified time: 2018-10-16 10:10:02


from __future__ import absolute_import, division, print_function

import operator
import os
import warnings
from copy import deepcopy

import astropy.units as units
import numpy as np

import marvin
import marvin.api.api
import marvin.core.exceptions
import marvin.core.marvin_pickle
import marvin.utils.general
import marvin.utils.plot.map
from marvin.utils.datamodel.dap.base import Property
from marvin.utils.general.general import add_doc

from .base_quantity import QuantityMixIn


try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None


class Map(units.Quantity, QuantityMixIn):
    """Describes 2D array object with addtional features.

    A general-use `~astropy.units.Quantity`-powered 2D array with features
    such as units, inverse variance, mask, quick acces to statistics
    methods, and plotting. While `.Map` can be used for any 2D array, it is
    mostly intended to represent an extension in a DAP MAPS file (normally
    known in MaNGA as a map). Unlike a `~marvin.tools.maps.Maps` object,
    which contains all the information from a DAP MAPS file, this class
    represents only one of the multiple 2D maps contained
    within. For instance, `~marvin.tools.maps.Maps` may contain emission line
    maps for multiple channels. A `.Map` would be, for example, the map for
    ``emline_gflux`` and channel ``ha_6564``.

    A `.Map` is normally initialised from a `~marvin.tools.maps.Maps` by
    calling the `~marvin.tools.maps.Maps.getMap` method. It can be
    initialialised directly using the `.Map.from_maps` classmethod.

    Parameters:
        array (array-like):
            The 2-D array contianing the spectrum.
        unit (astropy.unit.Unit, optional):
            The unit of the spectrum.
        scale (float, optional):
            The scale factor of the spectrum value.
        ivar (array-like, optional):
            The inverse variance array for ``value``.
        mask (array-like, optional):
            The mask array for ``value``.
        binid (array-like, optional):
            The associated binid map.
        pixmask_flag (str):
            The maskbit flag to be used to convert from mask bits to labels
            (e.g., MANGA_DAPPIXMASK).

    """

    def __new__(cls, array, unit=None, scale=1, ivar=None, mask=None,
                binid=None, pixmask_flag=None, dtype=None, copy=True):

        if scale is not None:
            unit = units.CompositeUnit(unit.scale * scale, unit.bases, unit.powers)

        obj = units.Quantity(np.array(array), unit=unit, dtype=dtype, copy=copy)
        obj = obj.view(cls)
        obj._set_unit(unit)

        obj._maps = None
        obj._datamodel = None

        obj.ivar = np.array(ivar) if ivar is not None else None
        obj.mask = np.array(mask) if mask is not None else None
        obj.binid = np.array(binid) if binid is not None else None

        obj.pixmask_flag = pixmask_flag

        return obj

    def __repr__(self):

        if self.datamodel is not None:
            full = self.datamodel.full()
        else:
            full = None

        return ('<Marvin Map (property={0!r})>\n{1} {2}'.format(full, self.value,
                                                                self.unit.to_string()))

    def __getitem__(self, sl):

        new_obj = super(Map, self).__getitem__(sl)

        if type(new_obj) is not type(self):
            new_obj = self._new_view(new_obj)

        new_obj._set_unit(self.unit)

        new_obj.ivar = self.ivar.__getitem__(sl) if self.ivar is not None else self.ivar
        new_obj.mask = self.mask.__getitem__(sl) if self.mask is not None else self.mask
        new_obj.binid = self.binid.__getitem__(sl) if self.binid is not None else self.binid

        return new_obj

    def __deepcopy__(self, memo):

        new_map = Map(array=deepcopy(self.value, memo),
                      unit=deepcopy(self.unit, memo),
                      ivar=deepcopy(self.ivar, memo),
                      mask=deepcopy(self.mask, memo),
                      binid=deepcopy(self.binid, memo))

        new_map._maps = deepcopy(self._maps, memo)

        # TODO: temporary fix because the datamodel does deepcopy. Fix.
        new_map._datamodel = self._datamodel

        new_map.manga_target1 = deepcopy(self.manga_target1)
        new_map.manga_target2 = deepcopy(self.manga_target2)
        new_map.manga_target3 = deepcopy(self.manga_target3)
        new_map.target_flags = deepcopy(self.target_flags)

        new_map.pixmask_flag = deepcopy(self.pixmask_flag)

        return new_map

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self._datamodel = getattr(obj, '_datamodel', None)
        self._maps = getattr(obj, '_maps', None)

        self.ivar = getattr(obj, 'ivar', None)
        self.mask = getattr(obj, 'mask', None)
        self.binid = getattr(obj, 'binid', None)
        self.pixmask_flag = getattr(obj, 'pixmask_flag', None)

        self._set_unit(getattr(obj, 'unit', None))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        ivar = self._ivar_ufunc(ufunc, *inputs, **kwargs)

        return EnhancedMap(
            value=ufunc(self.value),
            ivar=ivar,
            mask=self.mask,
            unit=self.unit,  # unit
            datamodel=self._datamodel,
            pixmask_flag=self.pixmask_flag,
            copy=True,
        )

    def _ivar_ufunc(self, ufunc, *inputs, **kwargs):

        if ufunc in [np.log10]:
            compute_ivar = getattr(self, '_{}_ivar'.format(ufunc.__name__))
            ivar = compute_ivar(ivar=inputs[0].ivar, value=inputs[0].value, **kwargs)

        else:
            raise NotImplementedError('np.{0} is not implemented for {1}.'
                                      .format(ufunc.__name__, self.__class__.__name__))

        return ivar

    def getMaps(self):
        """Returns the associated `~marvin.tools.maps.Maps`."""

        if not self._maps:
            raise ValueError('this Map is not linked to any MAPS.')

        return self._maps

    @property
    def datamodel(self):
        """Returns the associated `~marvin.utils.datamodel.dap.Property`."""

        if not self._datamodel:
            raise ValueError('this Map does not have an associated Property.')

        return self._datamodel

    @classmethod
    def from_maps(cls, maps, prop, dtype=None, copy=True):
        """Initialise a `.Map` from a `.~marvin.tools.maps.Maps`."""

        import marvin.tools.maps

        assert isinstance(maps, marvin.tools.maps.Maps)
        assert isinstance(prop, Property)

        maps = maps
        datamodel = maps.datamodel

        assert prop.full() in datamodel, 'failed sanity check. Property does not match.'

        if maps.data_origin == 'file':
            value, ivar, mask = cls._get_map_from_file(maps, prop)
        elif maps.data_origin == 'db':
            value, ivar, mask = cls._get_map_from_db(maps, prop)
        elif maps.data_origin == 'api':
            value, ivar, mask = cls._get_map_from_api(maps, prop)

        # Gets the binid array for this property.
        if prop.name != 'binid':
            binid = maps.get_binid(prop)
        else:
            binid = None

        unit = prop.unit

        obj = cls(value, unit=unit, ivar=ivar, mask=mask, binid=binid,
                  dtype=dtype, copy=copy)

        obj._datamodel = prop
        obj._maps = maps

        obj.manga_target1 = maps.manga_target1
        obj.manga_target2 = maps.manga_target2
        obj.manga_target3 = maps.manga_target3
        obj.target_flags = maps.target_flags

        obj.pixmask_flag = prop.pixmask_flag

        return obj

    @staticmethod
    def _get_map_from_file(maps, prop):
        """Initialise the `.Map` from a `.~marvin.tools.maps.Maps` file."""

        if prop.channel is not None:
            channel_idx = prop.channel.idx
            value = maps.data[prop.name].data[channel_idx]
            ivar = maps.data[prop.name + '_ivar'].data[channel_idx] if prop.ivar else None
            mask = maps.data[prop.name + '_mask'].data[channel_idx] if prop.mask else None
        else:
            value = maps.data[prop.name].data
            ivar = maps.data[prop.name + '_ivar'].data if prop.ivar else None
            mask = maps.data[prop.name + '_mask'].data if prop.mask else None

        return value, ivar, mask

    @staticmethod
    def _get_map_from_db(maps, prop):
        """Initialise the `.Map` from the DB."""

        mdb = marvin.marvindb

        if not mdb.isdbconnected:
            raise marvin.core.exceptions.MarvinError('No db connected')

        if sqlalchemy is None:
            raise marvin.core.exceptions.MarvinError('sqlalchemy required to access the local DB.')

        assert prop.model is not None
        table = getattr(mdb.dapdb, prop.model)

        fullname_value = prop.db_column()
        value = mdb.session.query(getattr(table, fullname_value)).filter(
            table.file_pk == maps.data.pk).order_by(table.spaxel_index).all()

        shape = (int(np.sqrt(len(value))), int(np.sqrt(len(value))))
        value = np.array(value).reshape(shape).T

        ivar = None
        mask = None

        if prop.ivar:
            fullname_ivar = prop.db_column(ext='ivar')
            ivar = mdb.session.query(getattr(table, fullname_ivar)).filter(
                table.file_pk == maps.data.pk).order_by(table.spaxel_index).all()
            ivar = np.array(ivar).reshape(shape).T

        if prop.mask:
            fullname_mask = prop.db_column(ext='mask')
            mask = mdb.session.query(getattr(table, fullname_mask)).filter(
                table.file_pk == maps.data.pk).order_by(table.spaxel_index).all()
            mask = np.array(mask).reshape(shape).T

        return value, ivar, mask

    @staticmethod
    def _get_map_from_api(maps, prop):
        """Initialise the `.Map` from the API."""

        url = marvin.config.urlmap['api']['getmap']['url']

        url_full = url.format(
            **{'name': maps.plateifu,
               'property_name': prop.name,
               'channel': prop.channel.name if prop.channel else None,
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

        return value, ivar, mask

    def getSpaxel(self, **kwargs):
        """Returns a `~marvin.tools.spaxel.Spaxel`."""

        if not self._maps:
            raise ValueError('this Map does not have an associate parent Maps.')

        return self._maps.getSpaxel(**kwargs)

    def save(self, path, overwrite=False):
        """Pickle the map to a file.

        This method will fail if the map is associated to a Maps loaded
        from the db.

        Parameters:
            path (str):
                The path of the file to which the ``Map`` will be saved.
                Unlike for other Marvin Tools that derive from
                :class:`~marvin.tools.core.MarvinToolsClass`, ``path`` is
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
        """Restore a Map object from a pickled file.

        If ``delete=True``, the pickled file will be removed after it has been
        unplickled. Note that, for map objects instantiated from a Maps object
        with ``data_origin='file'``, the original file must exists and be
        in the same path as when the object was first created.
        """
        return marvin.core.marvin_pickle.restore(path, delete=delete)

    @property
    def masked(self):
        """Return a masked array using the recommended masks."""

        assert self.mask is not None, 'mask is None'

        default_params = self._datamodel.parent.get_default_plot_params()
        labels = default_params['default']['bitmasks']

        return np.ma.array(self.value, mask=self.pixmask.get_mask(labels, dtype=bool))

    @property
    def std(self):
        """The standard deviation of the measurement."""

        return self.error

    @staticmethod
    def _add_ivar(ivar1, ivar2, *args, **kwargs):
        return 1. / ((1. / ivar1 + 1. / ivar2))

    @staticmethod
    def _mul_ivar(ivar1, ivar2, value1, value2, value12):
        with np.errstate(divide='ignore', invalid='ignore'):
            sig1 = 1. / np.sqrt(ivar1)
            sig2 = 1. / np.sqrt(ivar2)
            sig12 = abs(value12) * ((sig1 / abs(value1)) + (sig2 / abs(value2)))
            ivar12 = 1. / sig12**2
        return ivar12

    @staticmethod
    def _pow_ivar(ivar, value, power):
        if ivar is None:
            return np.zeros(value.shape)
        else:
            sig = np.sqrt(1. / ivar)
            sig_out = value**power * power * sig * value
            ivar = 1 / sig_out**2.
            ivar[np.isnan(ivar) | np.isinf(ivar)] = 0
            return ivar

    @staticmethod
    def _log10_ivar(ivar, value):
        return np.log10(np.e) * ivar**-0.5 / value

    @staticmethod
    def _unit_propagation(unit1, unit2, op):
        ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}

        if op in ['*', '/']:
            unit12 = ops[op](unit1, unit2)
        else:
            if unit1 == unit2:
                unit12 = unit1
            else:
                warnings.warn('Units do not match for map arithmetic.', UserWarning)
                unit12 = None

        return unit12

    def _arith(self, term2, op):
        if isinstance(term2, (Map, EnhancedMap)):
            map_out = self._map_arith(term2, op)
        else:
            map_out = self._scalar_arith(term2, op)
        return map_out

    def _scalar_arith(self, scalar, op):
        """Do map arithmetic and correctly handle map attributes."""

        ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}

        ivar_func = {'+': lambda ivar, c: ivar, '-': lambda ivar, c: ivar,
                     '*': lambda ivar, c: ivar / c**2, '/': lambda ivar, c: ivar * c**2}

        with np.errstate(divide='ignore', invalid='ignore'):
            map_value = ops[op](self.value, scalar)

        map1_ivar = self.ivar if self.ivar is not None else np.zeros(self.shape)
        map_ivar = ivar_func[op](map1_ivar, scalar)

        map_mask = self.mask if self.mask is not None else np.zeros(self.shape, dtype=int)

        bad = np.isnan(map_value) | np.isinf(map_value)
        map_mask[bad] = map_mask[bad] | self.pixmask.labels_to_value('DONOTUSE')

        return EnhancedMap(value=map_value, unit=self.unit, ivar=map_ivar, mask=map_mask,
                           datamodel=self._datamodel, pixmask_flag=self.pixmask_flag, copy=True)

    def _map_arith(self, map2, op):
        """Do map arithmetic and correctly handle map attributes."""

        ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}

        assert self.shape == map2.shape, 'Cannot do map arithmetic on maps of different shapes.'

        ivar_func = {'+': self._add_ivar, '-': self._add_ivar,
                     '*': self._mul_ivar, '/': self._mul_ivar}

        with np.errstate(divide='ignore', invalid='ignore'):
            map12_value = ops[op](self.value, map2.value)

        map1_ivar = self.ivar if self.ivar is not None else np.zeros(self.shape)
        map2_ivar = map2.ivar if map2.ivar is not None else np.zeros(map2.shape)
        map12_ivar = ivar_func[op](map1_ivar, map2_ivar, self.value, map2.value, map12_value)
        map12_ivar[np.isnan(map12_ivar) | np.isinf(map12_ivar)] = 0

        map1_mask = self.mask if self.mask is not None else np.zeros(self.shape, dtype=int)
        map2_mask = map2.mask if map2.mask is not None else np.zeros(map2.shape, dtype=int)
        map12_mask = map1_mask | map2_mask

        bad = np.isnan(map12_value) | np.isinf(map12_value)
        map12_mask[bad] = map12_mask[bad] | self.pixmask.labels_to_value('DONOTUSE')

        map12_unit = self._unit_propagation(self.unit, map2.unit, op)

        return EnhancedMap(value=map12_value, unit=map12_unit, ivar=map12_ivar, mask=map12_mask,
                           datamodel=self._datamodel, pixmask_flag=self.pixmask_flag, copy=True)

    def __add__(self, map2):
        """Add two maps."""
        return self._arith(map2, '+')

    def __sub__(self, map2):
        """Subtract two maps."""
        return self._arith(map2, '-')

    def __mul__(self, map2):
        """Multiply two maps."""
        return self._arith(map2, '*')

    def __div__(self, map2):
        """Divide two maps."""
        return self._arith(map2, '/')

    def __truediv__(self, map2):
        """Divide two maps."""
        return self.__div__(map2)

    def __pow__(self, power):
        """Raise map to power.

        Parameters:
            power (float):
               Power to raise the map values.

        Returns:
            map (:class:`~marvin.tools.quantities.EnhancedMap` object)
        """
        value = self.value**power
        ivar = self._pow_ivar(self.ivar, self.value, power)
        unit = self.unit**power

        return EnhancedMap(value=value, unit=unit, ivar=ivar, mask=self.mask,
                           datamodel=self._datamodel, pixmask_flag=self.pixmask_flag, copy=True)

    def inst_sigma_correction(self):
        """Correct for instrumental broadening.

        Correct observed stellar or emission line velocity dispersion
        for instrumental broadening.
        """

        if self.datamodel.name == 'stellar_sigma':

            if 'MPL-4' in self._datamodel.parent.aliases:
                raise marvin.core.exceptions.MarvinError(
                    'Instrumental broadening correction not implemented for MPL-4.')
            elif 'MPL-6' in self._datamodel.parent.aliases:
                raise marvin.core.exceptions.MarvinError(
                    'The stellar sigma corrections in MPL-6 are unreliable. Please use MPL-7.')

            map_corr = self.getMaps()['stellar_sigmacorr']

        elif self.datamodel.name == 'emline_gsigma':
            map_corr = self.getMaps().getMap(property_name='emline_instsigma',
                                             channel=self.datamodel.channel.name)

        else:
            raise marvin.core.exceptions.MarvinError(
                'Cannot correct {0} for instrumental broadening.'.format(self.datamodel.full()))

        sigcorr = (self**2 - map_corr**2)**0.5
        sigcorr.ivar = (sigcorr.value / self.value) * self.ivar
        sigcorr.ivar[self.ivar == 0] = 0
        sigcorr.ivar[map_corr.value >= self.value] = 0
        sigcorr.value[map_corr.value >= self.value] = 0

        sigcorr._show_datamodel = True

        return sigcorr

    def specindex_correction(self):
        """Correct spectral index measurements for velocity dispersion.
        """
        if self.datamodel.name == 'specindex':

            if 'MPL-4' in self._datamodel.parent.aliases:
                raise marvin.core.exceptions.MarvinError(
                    'Velocity dispersion correction not implemented for MPL-4.')

            corr_name = '_'.join((self.datamodel.name, 'corr', self.datamodel.channel.name))
            map_corr = self.getMaps()[corr_name]

        else:
            raise marvin.core.exceptions.MarvinError(
                'Cannot correct {0} for velocity dispersion.'.format(self.datamodel.full()))

        if self.unit in ['Angstrom', '']:
            specindex_corr = self * map_corr

        elif self.unit == 'mag':
            specindex_corr = self + map_corr

        else:
            raise marvin.core.exceptions.MarvinError(
                'Specindex unit must be "Angstrom", "mag", or "" (i.e., dimensionless)')

        specindex_corr._show_datamodel = True

        return specindex_corr

    @add_doc(marvin.utils.plot.map.plot.__doc__)
    def plot(self, *args, **kwargs):
        return marvin.utils.plot.map.plot(dapmap=self, *args, **kwargs)


class EnhancedMap(Map):
    """Creates a Map that has been modified."""

    def __new__(cls, value, unit, *args, **kwargs):
        ignore = ['datamodel']
        [kwargs.pop(it) for it in ignore if it in kwargs]
        return super(EnhancedMap, cls).__new__(cls, value, unit=unit, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        self._datamodel = kwargs.get('datamodel', None)
        self._show_datamodel = False

    def __repr__(self):
        return ('<Marvin EnhancedMap>\n{0!r} {1!r}').format(self.value, self.unit.to_string())

    def __deepcopy__(self, memo):
        return EnhancedMap(value=deepcopy(self.value, memo), unit=deepcopy(self.unit, memo),
                           ivar=deepcopy(self.ivar, memo), mask=deepcopy(self.mask, memo),
                           pixmask_flag=deepcopy(self.pixmask_flag, memo), copy=True)

    def _init_map_from_maps(self):
        raise AttributeError("'EnhancedMap' has no attribute '_init_map_from_maps'.")

    def _get_from_file(self):
        raise AttributeError("'EnhancedMap' has no attribute '_get_from_file'.")

    def _get_from_db(self):
        raise AttributeError("'EnhancedMap' has no attribute '_get_from_db'.")

    def _get_from_api(self):
        raise AttributeError("'EnhancedMap' has no attribute '_get_from_api'.")

    def inst_sigma_correction(self):
        """Override Map.inst_sigma_correction with AttributeError."""
        raise AttributeError("'EnhancedMap' has no attribute 'inst_sigma_correction'.")

    @property
    def datamodel(self):
        if self._show_datamodel:
            return self._datamodel
        else:
            return None

    @datamodel.setter
    def datamodel(self, value):
        self._datamodel = value
        self._show_datamodel = True
