#!/usr/bin/env python
# encoding: utf-8
#
# spectrum.py
#
# Licensed under a 3-clause BSD license.

# Revision history:
#     13 Apr 2016 J. Sánchez-Gallego
#       Initial version


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

from astropy.units import Quantity, dimensionless_unscaled


class Spectrum(Quantity):
    """A class representing an spectrum with extra functionality.

    Parameters:
        flux (array-like):
            The 1-D array contianing the spectrum.
        unit (astropy.unit.Unit, optional):
            The unit of the flux spectrum.
        scale (float, optional):
            The scale factor of the spectrum flux.
        ivar (array-like, optional):
            The inverse variance array for ``spectrum``. Must have the same
            number of elements.
        mask (array-like, optional):
            The mask array for ``spectrum``. Must have the same number of
            elements.
        wavelength (array-like, optional):
            The wavelength solution for ``spectrum``. Must have the same number
            of elements.
        wavelength_unit (astropy.unit.Unit, optional):
            The units of the wavelength solution.

    Returns:
        spectrum:
            An astropy Quantity-like object that contains the spectrum, as well
            as inverse variance, mask, and wavelengh (itself a Quantity array).

    """

    def __new__(cls, flux, scale=1, unit=dimensionless_unscaled,
                wavelength_unit=dimensionless_unscaled, ivar=None, mask=None,
                wavelength=None, dtype=None, copy=True, **kwargs):

        flux = np.array(flux) * scale

        obj = Quantity(flux, unit=unit, dtype=dtype, copy=copy)
        obj = obj.view(cls)
        obj._set_unit(unit)

        obj.ivar = (np.array(ivar) / (scale ** 2)) if ivar is not None else None
        obj.mask = np.array(mask) if mask is not None else None

        if wavelength is None:
            obj.wavelength = None
        else:
            obj.wavelength = np.array(wavelength)
            if wavelength_unit:
                obj.wavelength *= wavelength_unit

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.ivar = getattr(obj, 'ivar', None)
        self.mask = getattr(obj, 'mask', None)
        self.wavelength = getattr(obj, 'wavelength', None)

    def __getitem__(self, sl):

        new_obj = super(Spectrum, self).__getitem__(sl)

        if type(new_obj) is not type(self):
            new_obj = self._new_view(new_obj)

        new_obj._set_unit(self.unit)

        new_obj.ivar = self.ivar.__getitem__(sl) if self.ivar is not None else self.ivar
        new_obj.mask = self.mask.__getitem__(sl) if self.mask is not None else self.mask
        new_obj.wavelength = self.wavelength.__getitem__(sl) \
            if self.wavelength is not None else self.wavelength

        return new_obj

    @property
    def error(self):
        """The standard deviation of the measurement."""

        if self.ivar is None:
            return None

        np.seterr(divide='ignore')

        return np.sqrt(1. / self.ivar) * self.unit

    @property
    def snr(self):
        """The signal to noise of the measurement."""

        if self.ivar is None:
            return None

        return np.abs(self.value * np.sqrt(self.ivar))

    @property
    def masked(self):
        """Returns a masked array."""

        return np.ma.array(self.value, mask=self.mask > 0)

    def plot(self, array='flux', xlim=None, ylim=(0, None), mask_color=None,
             xlabel=None, ylabel=None, figure=None, return_figure=False, **kwargs):
        """Plots a spectrum using matplotlib.

        Returns a |axes|_ object with a representation of this spectrum.
        The returned ``axes`` object can then be showed, modified, or saved to
        a file. If running Marvin from an iPython console and
        `matplotlib.pyplot.ion()
        <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.ion>`_,
        the plot will be displayed interactivelly.

        Parameters:
            array ({'flux', 'ivar', 'mask'}):
                The array to display, defaults to the internal spectrum with
                which the object was initialised.
            xlim,ylim (tuple-like or None):
                The range to display for the x- and y-axis, respectively,
                defined as a tuple of two elements ``[xmin, xmax]``. If
                the range is ``None``, the range for the axis will be set
                automatically by matploltib. If ``Spectrum.wavelength`` is
                defined, the range in the x-axis must be defined as a
                wavelength range. Default for ylim is (0, None), which cuts
                off negative values but lets the maximum float.
            xlabel,ylabel (str or None):
                The axis labels to be passed to the plot. If ``xlabel=None``
                and ``Spectrum.wavelength_unit`` is defined, those units will
                be used, after being properly formatted for Latex display.
                If ``ylabel=None``, the y-axis label will be automatically
                defined base on the type of input array.
            mask_color (matplotlib valid color or None):
                If set and ``Spectrum.mask`` is defined, the elements of
                ``array`` with ``mask`` will be coloured using that value.
                More information about `matplotlib colours
                <http://matplotlib.org/api/colors_api.html>`_.
            figure (matplotlib Figure object or None):
                The matplotlib figure object from which the axes must be
                created. If ``figure=None``, a new figure will be created.
            return_figure (bool):
                If ``True``, the matplotlib Figure object used will be returned
                along with the axes object.
            kwargs (dict):
                Any other keyword argument that will be passed to
                `matplotlib.pyplot.plot
                <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_.

        Returns:
            ax:
                The `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_
                object containing the plot representing the spectrum. If
                ``return_figure=True``, a tuple will be returned of the form
                ``(ax, fig)``.

        Example:

          >>> spectrum = Spectrum(np.arange(100), wavelength=np.arange(100)*0.1)
          >>> ax = spectrum.plot(xrange=[5, 7])
          >>> ax.show()

        We can change the range of the axes after the object has been created.

          >>> ax.set_xlim(3, 8)
          >>> ax.show()

        .. |axes| replace:: matplotlib.axes
        .. _axes: http://matplotlib.org/api/axes_api.html

        """

        array = array.lower()
        validSpectrum = ['flux', 'ivar', 'mask']
        assert array in validSpectrum, 'array must be one of {0!r}'.format(validSpectrum)

        if array == 'flux':
            data = self.value
            unit = 'Flux [{0}]'.format(self.unit.to_string('latex_inline'))
        elif array == 'ivar':
            assert self.ivar is not None, 'ivar is None'
            data = self.ivar
            unit = 'Inverse variance [{0}]'.format(((1 / self.unit) ** 2).to_string('latex_inline'))
        elif array == 'mask':
            assert self.mask is not None, 'mask is None'
            data = self.mask
            unit = ''

        xaxis = self.wavelength if self.wavelength is not None else np.arange(len(self))

        fig = plt.figure() if figure is None else figure
        ax = fig.add_subplot(111)

        ax.plot(xaxis, data, **kwargs)

        # This does not work very well for small ranges of masked elements.
        # Probably requires some rethinking.
        if mask_color is not None:
            mask_indices = np.where(self.mask > 0)
            kwargs['color'] = mask_color
            ax.plot(xaxis[mask_indices], data[mask_indices], **kwargs)

        if xlim is not None:
            assert len(xlim) == 2
            ax.set_xlim(*xlim)

        if ylim is not None:
            assert len(ylim) == 2
            ax.set_ylim(*ylim)

        if xlabel is None:
            if self.wavelength is not None:
                xlabel = 'Wavelength [{0}]'.format(self.wavelength.unit.to_string('latex_inline'))
            else:
                xlabel = ''

        if ylabel is None:
            ylabel = unit

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if return_figure:
            return (ax, fig)
        else:
            return ax
