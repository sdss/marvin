#!/usr/bin/env python
# encoding: utf-8
#
# analysis_props.py
#
# Created by José Sánchez-Gallego on 13 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

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

    """

    def __new__(cls, value, unit=dimensionless_unscaled, scale=1,
                ivar=None, mask=None, dtype=None, copy=True):

        obj = Quantity(value, unit=unit, dtype=dtype, copy=copy)
        obj = obj.view(cls)
        obj._set_unit(unit)

        obj.ivar = ivar
        obj.mask = mask

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.ivar = getattr(obj, 'ivar', None)
        self.mask = getattr(obj, 'mask', None)
