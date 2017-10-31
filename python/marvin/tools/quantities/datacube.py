#!/usr/bin/env python
# encoding: utf-8
#
# datacube.py
#
# Created by José Sánchez-Gallego on 4 Oct 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from astropy import units

from marvin.utils.general.general import _sort_dir


class DataCube(units.Quantity):
    """A `~astropy.units.Quantity`-powered representation of a 3D data cube.

    A `DataCube` represents a 3D array in which two of the dimensions
    correspond to the spatial direction, with the third one being the
    spectral direction.

    Parameters:
        value (`~numpy.ndarray`):
            A 3-D array with the value of the quantity measured. The first
            axis of the array must be the wavelength dimension.
        wavelength (`~numpy.ndarray`):
            A 1-D array with the wavelenth of each spectral measurement. It
            must have the same length as the spectral dimesion of ``value``.
        unit (`~astropy.units.Unit`):
            An `astropy unit <astropy.units.Unit>` with the units for
            ``value``.
        scale (float):
            The scale factor of the spectrum value.
        wavelength_unit (astropy.unit.Unit, optional):
            The units of the wavelength solution. Defaults to Angstrom.
        ivar (`~numpy.ndarray`):
            An array with the same shape as ``value`` contianing the associated
            inverse variance.
        mask (`~numpy.ndarray`):
            Same as ``ivar`` but for the associated
            :ref:`bitmask <marvin-bitmasks>`.
        kwargs (dict):
            Keyword arguments to be passed to `~astropy.units.Quantity` when
            it is initialised.

    """

    def __new__(cls, value, wavelength, scale=None, unit=units.dimensionless_unscaled,
                wavelength_unit=units.Angstrom, ivar=None, mask=None, dtype=None,
                copy=True, **kwargs):

        # If the scale is defined, creates a new composite unit with the input scale.
        if scale is not None:
            unit = units.CompositeUnit(unit.scale * scale, unit.bases, unit.powers)

        assert wavelength is not None, 'a valid wavelength array is required'

        assert isinstance(value, np.ndarray) and value.ndim == 3, 'value must be a 3D array.'
        assert isinstance(wavelength, np.ndarray) and wavelength.ndim == 1, \
            'wavelength must be a 1D array.'
        assert len(wavelength) == value.shape[0], \
            'wavelength and value spectral dimensions do not match'

        if ivar is not None:
            assert isinstance(ivar, np.ndarray) and ivar.shape == value.shape, 'invalid ivar shape'
            assert isinstance(mask, np.ndarray) and mask.shape == value.shape, 'invalid mask shape'

        obj = units.Quantity(value, unit=unit, **kwargs)
        obj = obj.view(cls)
        obj._set_unit(unit)

        assert wavelength is not None, 'invalid wavelength'

        if isinstance(wavelength, units.Quantity):
            obj.wavelength = wavelength
        else:
            obj.wavelength = np.array(wavelength) * wavelength_unit

        obj.ivar = np.array(ivar) if ivar is not None else None
        obj.mask = np.array(mask) if mask is not None else None

        return obj

    def __getitem__(self, sl):

        new_obj = super(DataCube, self).__getitem__(sl)

        if type(new_obj) is not type(self):
            new_obj = self._new_view(new_obj)

        new_obj._set_unit(self.unit)

        new_obj.ivar = self.ivar.__getitem__(sl) if self.ivar is not None else self.ivar
        new_obj.mask = self.mask.__getitem__(sl) if self.mask is not None else self.mask

        return new_obj

    def __dir__(self):

        return_list = _sort_dir(self, DataCube)
        return_list += ['value']

        return return_list

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.wave = getattr(obj, 'wave', None)
        self.ivar = getattr(obj, 'ivar', None)
        self.mask = getattr(obj, 'mask', None)

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
