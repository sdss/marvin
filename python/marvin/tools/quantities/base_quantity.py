#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-10-31
# @Filename: base_quantity.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-23 16:42:56


from __future__ import absolute_import, division, print_function

import warnings

import astropy.units as units
import numpy as np

import marvin.core.exceptions
from marvin.utils.general import maskbit
from marvin.utils.general.general import _sort_dir


class QuantityMixIn(object):
    """A MixIn that provides common functionalities to Quantity classes."""

    def __dir__(self):

        return_list = _sort_dir(self, self.__class__)
        return_list += ['value']

        return return_list

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
            return np.ma.array(self.value, mask=(self.mask > 0))

        labels = pixmask.schema.label.tolist()
        if 'DONOTUSE' in labels and 'NOCOV' in labels:
            return np.ma.array(self.value, mask=self.pixmask.get_mask(['DONOTUSE', 'NOCOV']) > 0)
        elif 'DONOTUSE' in labels:
            return np.ma.array(self.value, mask=self.pixmask.get_mask('DONOTUSE') > 0)
        else:
            return np.ma.array(self.value, mask=(self.mask > 0))

    @property
    def error(self):
        """Compute the standard deviation of the measurement."""

        if hasattr(self, '_std') and self._std is not None:
            return self._std

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
