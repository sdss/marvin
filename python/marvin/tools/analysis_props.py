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


class AnalysisProperty(marvin.core.core.Dotable):
    """A class describing a property with a value and additional information.

    This class is intended for internal use, not to be initialisided by the
    user. The class is designed to mimic the DAP MAPS structure, with
    `category` reflecting the `name` of the header, and name the channel.
    However, the class is flexible and can be used to define any scalar
    property. For each value, an `ivar` and a `mask` value.

    Parameters:
        category (str):
            A string with the category to which this property belongs (e.g.,
            `emline_gflux`, `specindex`, etc).
        name (str):
            The name of the property (e.g., `ha`, `nii`, etc).
        value (float):
            The value of the property.
        unit (str or None):
            The units of `value`.
        ivar (float or None):
            The inverse variance associated with `value`, or `None` if not
            defined.
        mask (bool):
            Whether the value is masked or not.

    """

    def __init__(self, category, name, value, unit=None, ivar=None, mask=True):

        self['category'] = category
        self['name'] = name
        self['value'] = value
        self['unit'] = unit
        self['ivar'] = ivar
        self['mask'] = bool(mask)

        # super(AnalisisProperty, self).__init__()

    def __repr__(self):

        return ('<AnalysisProperty ({0.category}, {0.name}, value={0.value} '
                'ivar={0.ivar}, mask={0.mask!r})>'.format(self))
