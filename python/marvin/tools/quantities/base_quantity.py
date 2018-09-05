#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-10-31
# @Filename: base_quantity.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-08-12 13:51:26


from __future__ import absolute_import, division, print_function

import warnings

import astropy.units as units
import numpy

import marvin.core.exceptions
from marvin.tools.spaxel import Spaxel
from marvin.utils.general import maskbit
from marvin.utils.general.general import _sort_dir


class BinInfo(object):
    """Provides information about the bin associated with this quantity."""

    def __init__(self, spaxel=None, parent=None, datamodel=None):

        self._spaxel = spaxel
        self._parent = parent
        self._datamodel = datamodel

    def __repr__(self):

        return '<BinInfo (binid={0.binid}, n_spaxels={1})>'.format(
            self, numpy.sum(self.binid_mask))

    @property
    def binid_map(self):
        """Returns the binid Map associated to this quantity."""

        return self._parent.get_binid(self._datamodel)

    @property
    def binid(self):
        """Returns the binid associated to this quantity and spaxel."""

        return int(self.binid_map[self._spaxel.y, self._spaxel.x].value)

    @property
    def binid_mask(self):
        """Returns a mask of the spaxels with the same binid."""

        return self.binid_map.value == self.binid

    @property
    def is_binned(self):
        """Returns `True`` if the parent object is binned."""

        return self._parent.is_binned()

    def get_bin_spaxels(self, lazy=True):
        """Returns a list of the spaxels associated with this bin.

        Parameters
        ----------
        lazy : bool
            If ``True``, the spaxels returned will be lazy loaded. Spaxels
            can be fully loaded by calling their `~.Spaxel.load` method.

        Returns
        -------
        spaxels : list
            A list of all the `.Spaxel` instances associated with this
            quantity binid.

        """

        if self.binid < 0:
            raise marvin.core.exceptions.MarvinError(
                'coordinates ({}, {}) do not correspond to a valid binid.'.format(self._spaxel.x,
                                                                                  self._spaxel.y))

        spaxel_coords = zip(*numpy.where(self.binid_map.value == self.binid))
        spaxels = []

        for ii, jj in spaxel_coords:
            spaxels.append(Spaxel(x=jj, y=ii, plateifu=self._spaxel.plateifu,
                                  release=self._spaxel.release, cube=self._spaxel._cube,
                                  maps=self._spaxel._maps, modelcube=self._spaxel._modelcube,
                                  bintype=self._spaxel.bintype, template=self._spaxel.template,
                                  lazy=lazy))

        return spaxels


class QuantityMixIn(object):
    """A MixIn that provides common functionalities to Quantity classes."""

    def __dir__(self):

        return_list = _sort_dir(self, self.__class__)
        return_list += ['value']

        return return_list

    def _init_bin(self, spaxel=None, datamodel=None, parent=None):

        self.bin = BinInfo(spaxel=spaxel, datamodel=datamodel, parent=parent)

    @property
    def pixmask(self):
        """Maskbit instance for the pixmask flag.

        See :ref:`marvin-utils-maskbit` for documentation and
        `~marvin.utils.general.maskbit.Maskbit` for API reference.

        """

        assert self.pixmask_flag, 'pixmask flag not set'

        pixmask = maskbit.Maskbit(self.pixmask_flag)
        pixmask.mask = self.mask

        return pixmask

    @property
    def masked(self):
        """Return a masked array.

        If the `~QuantityMixIn.pixmask` is set, and the maskbit contains the
        ``DONOTUSE`` and ``NOCOV`` labels, the returned array will be masked
        for the values containing those bits. Otherwise, all values where the
        mask is greater than zero will be masked.

        """

        assert self.mask is not None, 'mask is not set'

        try:
            pixmask = self.pixmask
        except AssertionError:
            warnings.warn('pixmask not set. Applying full mask.',
                          marvin.core.exceptions.MarvinUserWarning)
            return numpy.ma.array(self.value, mask=(self.mask > 0))

        labels = pixmask.schema.label.tolist()
        if 'DONOTUSE' in labels and 'NOCOV' in labels:
            return numpy.ma.array(self.value,
                                  mask=self.pixmask.get_mask(['DONOTUSE', 'NOCOV']) > 0)
        elif 'DONOTUSE' in labels:
            return numpy.ma.array(self.value, mask=self.pixmask.get_mask('DONOTUSE') > 0)
        else:
            return numpy.ma.array(self.value, mask=(self.mask > 0))

    @property
    def error(self):
        """Compute the standard deviation of the measurement."""

        if hasattr(self, '_std') and self._std is not None:
            return self._std

        if self.ivar is None:
            return None

        numpy.seterr(divide='ignore')

        return numpy.sqrt(1. / self.ivar) * self.unit

    @property
    def snr(self):
        """Return the signal-to-noise ratio for each spaxel in the map."""

        return numpy.abs(self.value * numpy.sqrt(self.ivar))

    def descale(self):
        """Returns a copy of the object in which the scale is unity.

        Example:

            >>> dc.unit
            Unit("1e-17 erg / (Angstrom cm2 s spaxel)")
            >> dc[100, 15, 15]
            <DataCube 0.270078063011169 1e-17 erg / (Angstrom cm2 s spaxel)>
            >>> dc_descaled = dc.descale()
            >>> d_descaled.unit
            Unit("Angstrom cm2 s spaxel")
            >>> dc[100, 15, 15]
            <DataCube 2.70078063011169e-18 erg / (Angstrom cm2 s spaxel)>

        """

        if self.unit.scale == 1:
            return self

        value_descaled = self.value * self.unit.scale
        value_unit = units.CompositeUnit(1, self.unit.bases, self.unit.powers)

        if self.ivar is not None:
            ivar_descaled = self.ivar / (self.unit.scale ** 2)
        else:
            ivar_descaled = None

        return self.__class__(value_descaled, self.wavelength, unit=value_unit,
                              ivar=ivar_descaled, mask=self.mask)
