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
import marvin.core.core


__ALL__ = ('DictOfProperties', 'AnalysisProperty')


class DictOfProperties(marvin.core.core.FuzzyDict):
    """A dotable fuzzy dictionary to list a groups of AnalysisProperties."""

    pass


class AnalysisProperty(object):
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

    """

    def __init__(self, prop, value, ivar=None, mask=None):

        self.property = prop

        self.unit = self.property.unit

        self.value = value * self.property.scale * self.unit
        self.ivar = (ivar / (self.property.scale ** 2)) if ivar else None
        self.mask = mask

        self.name = self.property.name
        self.channel = self.property.channel

        self.description = self.property.description

    @property
    def error(self):
        """The standard deviation of the measurement."""

        if self.ivar is None:
            return None

        if self.ivar == 0:
            return np.inf

        return np.sqrt(1. / self.ivar) * self.unit

    @property
    def sn(self):
        """The signal to noise of the measurement."""

        if self.ivar is None:
            return None

        return (self.value * np.sqrt(self.ivar)).value

    def __repr__(self):

        return ('<AnalysisProperty (name={0.name!r}, channel={0.channel!r}, value={0.value} '
                'ivar={0.ivar}, mask={0.mask})>'.format(self))
