#!/usr/bin/env python
# encoding: utf-8
#
# analysis_props.py
#
# Created by José Sánchez-Gallego on 13 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import marvin.core.core


__all__ = ('DictOfProperties', 'AnalysisProperty')


class DictOfProperties(marvin.core.core.DotableCaseInsensitive):
    """A dotable dictionary to list a groups of AnalysisProperty objects."""

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
        name (str):
            A string with the property name to which this property belongs
            (e.g., ``emline_gflux``, ``specindex``, etc).
        channel (str or None):
            The name of the property (e.g., ``ha_6564``, ``nii_6585``, etc).
        value (float):
            The value of the property.
        unit (str or None):
            The units of ``value``.
        ivar (float or None):
            The inverse variance associated with ``value``, or ``None`` if not
            defined.
        mask (bool):
            The value of the mask for this value.
        description (str):
            A string describing the property.

    """

    def __init__(self, name, channel, value, unit=None, ivar=None, mask=None,
                 description=''):

        self.name = name
        self.channel = channel
        self.value = value
        self.unit = unit
        self.ivar = ivar
        self.mask = mask
        self.description = description

    def __repr__(self):

        return ('<AnalysisProperty (name={0.name}, channels={0.channel}, value={0.value} '
                'ivar={0.ivar}, mask={0.mask})>'.format(self))
