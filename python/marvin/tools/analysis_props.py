#!/usr/bin/env python
# encoding: utf-8
#
# analysis_props.py
#
# Created by José Sánchez-Gallego on 13 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import marvin.utils.general.structs

from astropy.units import Quantity


__ALL__ = ('DictOfProperties', 'AnalysisProperty')


class DictOfProperties(marvin.utils.general.structs.FuzzyDict):
    """A dotable fuzzy dictionary to list a groups of AnalysisProperties."""

    pass


class AnalysisProperty(Quantity):
    """A class describing a property with a value and additional information.

    This class is intended for internal use, not to be initialisided by the
    user. The class is designed to mimic the DAP MAPS structure, with
    ``property_name`` reflecting the name of the proery being represented
    (e.g., ``'emline_gflux'``) and ``channel``, if any, the channel to which
    the values make reference. However, the class is flexible and can be used
    to define any scalar property. For each value, an ``ivar``, ``mask``, and
    ``unit`` can be defined.

    Parameters:
        prop (DAP Property object):
            A Property object with model information about the data.
        value (float):
            The value of the property.
        ivar (float or None):
            The inverse variance associated with ``value``, or ``None`` if not
            defined.
        mask (int or None):
            The mask value associated with ``value``, or ``None`` if not
            defined.
        quality_flag (Maskbit object):
        targeting_flags (list)
        mngtarg1 (Maskbit object)
        mngtarg2 (Maskbit object)
        mngtarg3 (Maskbit object)
        pixmask (Maskbit object)


    """

    def __new__(cls, prop, value, ivar=None, mask=None, dtype=None, copy=True):

        unit = prop.unit
        scale = prop.scale
        value = value * scale * unit

        obj = Quantity(value, unit=unit, dtype=dtype, copy=copy)
        obj = obj.view(cls)
        obj._set_unit(unit)

        obj.property = prop

        obj.ivar = (ivar / (obj.property.scale ** 2)) if ivar is not None else None
        obj.mask = mask

        obj.name = obj.property.name
        obj.channel = obj.property.channel

        obj.description = obj.property.description

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.name = getattr(obj, 'name', None)
        self.channel = getattr(obj, 'channel', None)
        self.description = getattr(obj, 'description', None)

        self.ivar = getattr(obj, 'ivar', None)
        self.mask = getattr(obj, 'mask', None)

    @property
    def error(self):
        """The standard deviation of the measurement."""

        if self.ivar is None:
            return None

        if self.ivar == 0:
            return np.inf

        return np.sqrt(1. / self.ivar) * self.unit

    @property
    def snr(self):
        """The signal to noise of the measurement."""

        if self.ivar is None:
            return None

        return np.abs(self.value * np.sqrt(self.ivar))

    def __repr__(self):

        return ('<AnalysisProperty (name={0.name!r}, channel={0.channel!r}, '
                'value={0} ivar={0.ivar}, mask={0.mask})>'.format(self))
