#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2016-06-13
# @Filename: analysis_props.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-22 01:52:54


from __future__ import absolute_import, division, print_function

from astropy.units import Quantity, dimensionless_unscaled

from .base_quantity import QuantityMixIn


__all__ = ['AnalysisProperty']


class AnalysisProperty(Quantity, QuantityMixIn):
    """A class describing a measurement with additional information.

    Represents a quantity with an associated unit and possibly an associated
    error and mask.

    Parameters:
        value (float):
            The value of the quantity.
        unit (astropy.unit.Unit, optional):
            The unit of the quantity.
        scale (float, optional):
            The scale factor of the quantity value.
        ivar (float or None):
            The inverse variance associated with ``value``, or ``None`` if not
            defined.
        mask (int or None):
            The mask value associated with ``value``, or ``None`` if not
            defined.
        pixmask_flag (str):
            The maskbit flag to be used to convert from mask bits to labels
            (e.g., MANGA_DAPPIXMASK).

    """

    def __new__(cls, value, unit=dimensionless_unscaled, scale=1,
                ivar=None, mask=None, pixmask_flag=None, dtype=None, copy=True):

        obj = Quantity(value, unit=unit, dtype=dtype, copy=copy)
        obj = obj.view(cls)
        obj._set_unit(unit)

        obj.ivar = ivar
        obj.mask = mask
        obj.pixmask_flag = pixmask_flag

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.ivar = getattr(obj, 'ivar', None)
        self.mask = getattr(obj, 'mask', None)

        self._set_unit(getattr(obj, 'unit', None))

    @property
    def std(self):
        """The standard deviation of the measurement."""

        return self.error
