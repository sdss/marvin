#!/usr/bin/env python
# encoding: utf-8
#
# anal_props.py
#
# Created by José Sánchez-Gallego on 13 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


class AnalisisProperty(object):
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
        ivar (float or None):
            The inverse variance associated with `value`, or `None` if not
            defined.
        mask (bool):
            Whether the value is masked or not.

    """

    def __init__(self, category, name, value, ivar=None, mask=True):

        self.category = category
        self.name = name
        self.value = value
        self.ivar = ivar
        self.mask = mask
