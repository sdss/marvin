#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-10-30
# @Filename: datacube.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-21 23:27:25


from __future__ import absolute_import, division, print_function

import numpy as np
from astropy import units

from .base_quantity import QuantityMixIn
from .spectrum import Spectrum


class DataCube(units.Quantity, QuantityMixIn):
    """A `~astropy.units.Quantity`-powered representation of a 3D data cube.

    A `DataCube` represents a 3D array in which two of the dimensions
    correspond to the spatial direction, with the third one being the
    spectral direction.

    Parameters:
        value (`~numpy.ndarray`):
            A 3-D array with the value of the quantity measured. The first
            axis of the array must be the wavelength dimension.
        wavelength (`~numpy.ndarray`):
            A 1-D array with the wavelength of each spectral measurement. It
            must have the same length as the spectral dimesion of ``value``.
        unit (`~astropy.units.Unit`):
            An `astropy unit <astropy.units.Unit>` with the units for
            ``value``.
        scale (float):
            The scale factor of the spectrum value.
        wavelength_unit (astropy.unit.Unit):
            The units of the wavelength solution. Defaults to Angstrom.
        redcorr (`~numpy.ndarray`):
            The reddenning correction, used by `.ModelCube.deredden`.
        ivar (`~numpy.ndarray`):
            An array with the same shape as ``value`` contianing the associated
            inverse variance.
        mask (`~numpy.ndarray`):
            Same as ``ivar`` but for the associated
            :ref:`bitmask <marvin-bitmasks>`.
        binid (`~numpy.ndarray`):
            The associated binid map for this datacube. Only set for DAP
            `~marvin.tools.modelcube.ModelCube` datacubes.
        pixmask_flag (str):
            The maskbit flag to be used to convert from mask bits to labels
            (e.g., MANGA_DRP3PIXMASK).
        kwargs (dict):
            Keyword arguments to be passed to `~astropy.units.Quantity` when
            it is initialised.

    """

    def __new__(cls, value, wavelength, scale=None, unit=units.dimensionless_unscaled,
                wavelength_unit=units.Angstrom, redcorr=None, ivar=None, mask=None,
                binid=None, pixmask_flag=None, dtype=None, copy=True, **kwargs):

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
        if mask is not None:
            assert isinstance(mask, np.ndarray) and mask.shape == value.shape, 'invalid mask shape'
        if binid is not None:
            assert (isinstance(binid, np.ndarray) and
                    binid.shape == value.shape[1:]), 'invalid binid shape'

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
        obj.binid = np.array(binid) if binid is not None else None

        obj.pixmask_flag = pixmask_flag

        if redcorr is not None:
            assert len(redcorr) == len(obj.wavelength), 'invalid length for redcorr.'
            obj.redcorr = np.array(redcorr)

        return obj

    def __getitem__(self, sl):

        if isinstance(sl, tuple):
            sl_wave = sl[0]
            sl_spatial = sl[1:]
        else:
            sl_wave = sl
            sl_spatial = None

        new_obj = super(DataCube, self).__getitem__(sl)

        if type(new_obj) is not type(self):
            new_obj = self._new_view(new_obj)

        new_obj._set_unit(self.unit)

        if self.ivar is not None:
            new_obj.ivar = self.ivar.__getitem__(sl)
        else:
            new_obj.ivar = self.ivar

        if self.mask is not None:
            new_obj.mask = self.mask.__getitem__(sl)
        else:
            new_obj.mask = self.mask

        if self.wavelength is not None:
            new_obj.wavelength = self.wavelength.__getitem__(sl_wave)
        else:
            new_obj.wavelength = self.wavelength

        if self.redcorr is not None:
            new_obj.redcorr = self.redcorr.__getitem__(sl_wave)
        else:
            new_obj.redcorr = self.redcorr

        if self.binid is not None:
            new_obj.binid = self.binid.__getitem__(sl_spatial)
        else:
            new_obj.binid = self.binid

        if new_obj.ndim == 1 and not np.isscalar(new_obj.wavelength.value):
            return Spectrum(new_obj.value, unit=new_obj.unit, wavelength=new_obj.wavelength,
                            ivar=new_obj.ivar, mask=new_obj.mask, pixmask_flag=self.pixmask_flag)

        return new_obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.wavelength = getattr(obj, 'wavelength', None)
        self.ivar = getattr(obj, 'ivar', None)
        self.mask = getattr(obj, 'mask', None)
        self.redcorr = getattr(obj, 'redcorr', None)
        self.binid = getattr(obj, 'binid', None)

        self.pixmask_flag = getattr(obj, 'pixmask_flag', None)

        self._set_unit(getattr(obj, 'unit', None))

    @property
    def std(self):
        """The standard deviation of the measurement."""

        return self.error

    def deredden(self, redcorr=None):
        """Returns the dereddened datacube.

        Parameters
        ----------
        redcorr : float or None
            The reddening correction to apply. If ``None``, defaults to the
            ``DataCube.redcorr``.

        Returns
        -------
        deredden : DataCube
            A `DataCube` with the flux and ivar corrected from reddening.

        Raises
        ------
        ValueError
            If ``redcorr=None`` and ``DataCube.redcorr=None``.

        """

        redcorr = redcorr if redcorr is not None else self.redcorr

        if redcorr is None:
            raise ValueError('no reddening correction specified.')

        assert len(redcorr) == len(self.wavelength), 'invalid length for redcorr.'

        new_value = (self.value.T * redcorr).T
        new_ivar = (self.ivar.T / redcorr**2).T

        new_obj = DataCube(new_value, self.wavelength, unit=self.unit,
                           redcorr=None, ivar=new_ivar, mask=self.mask,
                           binid=self.binid, pixmask_flag=self.pixmask_flag)

        return new_obj
