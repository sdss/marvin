#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 31, 2017
# @Filename: base_quantity.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import astropy.units as units

from marvin.utils.general.general import _sort_dir


class QuantityMixIn(object):
    """A MixIn that provides common functionalities to Quantity classes."""

    def __dir__(self):

        return_list = _sort_dir(self, self.__class__)
        return_list += ['value']

        return return_list

    @property
    def masked(self):
        """Return a masked array."""

        assert self.mask is not None, 'mask is None'

        return np.ma.array(self.value, mask=self.mask > 0)

    @property
    def error(self):
        """Compute the standard deviation of the measurement."""

        if self.ivar is None:
            return None

        np.seterr(divide='ignore')

        return np.sqrt(1. / self.ivar) * self.unit

    @property
    def snr(self):
        """Return the signal-to-noise ratio for each spaxel in the map."""

        return np.abs(self.value * np.sqrt(self.ivar))

    def descale(self):
        """Returns a copy of the object in which the scale is unity.

        Example:

            >>> dc.unit
            Unit("1e-17 erg / (Angstrom cm2 s spaxel)")
            >> dc[100, 15, 15]
            <DataCube 0.270078063011169 1e-17 erg / (Angstrom cm2 s spaxel)>
            >> dc_descaled = dc.descale()
            >> d_descaled.unit
            Unit("Angstrom cm2 s spaxel")
            >> dc[100, 15, 15]
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
