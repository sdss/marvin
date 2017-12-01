#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 30, 2017
# @Filename: spectrum.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import matplotlib.pyplot as plt

from astropy.units import CompositeUnit, Quantity, Angstrom

from .base_quantity import QuantityMixIn


class Spectrum(Quantity, QuantityMixIn):
    """A class representing an spectrum with extra functionality.

    Parameters:
        value (array-like):
            The 1-D array contianing the spectrum.
        wavelength (array-like, optional):
            The wavelength solution for ``value``.
        scale (float, optional):
            The scale factor of the spectrum value.
        unit (astropy.unit.Unit, optional):
            The unit of the spectrum.
        wavelength_unit (astropy.unit.Unit, optional):
            The units of the wavelength solution. Defaults to Angstrom.
        ivar (array-like, optional):
            The inverse variance array for ``value``.
        std (array-like, optional):
            The standard deviation associated with ``value``.
        mask (array-like, optional):
            The mask array for ``value``.
        kwargs (dict):
            Keyword arguments to be passed to `~astropy.units.Quantity` when
            it is initialised.

    Returns:
        spectrum:
            An astropy Quantity-like object that contains the spectrum, as well
            as inverse variance, mask, and wavelength (itself a Quantity array).

    """

    def __new__(cls, flux, wavelength, scale=None, unit=Angstrom,
                wavelength_unit=Angstrom, ivar=None, std=None,
                mask=None, dtype=None, copy=True, **kwargs):

        flux = np.array(flux)

        # If the scale is defined, creates a new composite unit with the input scale.
        if scale is not None:
            unit = CompositeUnit(unit.scale * scale, unit.bases, unit.powers)

        obj = Quantity(flux, unit=unit, dtype=dtype, copy=copy)
        obj = obj.view(cls)
        obj._set_unit(unit)

        obj.ivar = np.array(ivar) if ivar is not None else None
        obj.mask = np.array(mask) if mask is not None else None

        if std is not None:
            assert ivar is None, 'std and ivar cannot be used at the same time.'
            obj._std = np.array(std)

        assert wavelength is not None, 'invalid wavelength'

        if isinstance(wavelength, Quantity):
            obj.wavelength = wavelength
        else:
            obj.wavelength = np.array(wavelength) * wavelength_unit

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.ivar = getattr(obj, 'ivar', None)
        self._std = getattr(obj, '_std', None)
        self.mask = getattr(obj, 'mask', None)
        self.wavelength = getattr(obj, 'wavelength', None)

    def __getitem__(self, sl):

        new_obj = super(Spectrum, self).__getitem__(sl)

        if type(new_obj) is not type(self):
            new_obj = self._new_view(new_obj)

        new_obj._set_unit(self.unit)

        new_obj.ivar = self.ivar.__getitem__(sl) if self.ivar is not None else self.ivar
        new_obj._std = self._std.__getitem__(sl) if self._std is not None else self._std
        new_obj.mask = self.mask.__getitem__(sl) if self.mask is not None else self.mask
        new_obj.wavelength = self.wavelength.__getitem__(sl) \
            if self.wavelength is not None else self.wavelength

        return new_obj

    @property
    def error(self):
        """The standard deviation of the measurement."""

        if self._std is not None:
            return self._std

        if self.ivar is None:
            return None

        np.seterr(divide='ignore')

        return np.sqrt(1. / self.ivar) * self.unit

    @property
    def std(self):
        """The standard deviation of the measurement."""

        return self.error

    def plot(self, xlim=None, ylim=None, show_std=True, use_mask=True,
             n_sigma=1, xlabel='Wavelength', ylabel='Flux', show_units=True,
             plt_style='seaborn-darkgrid', figure=None, return_figure=False):
        """Plots the spectrum.

        Displays the spectrum showing, optionally, the :math:`n\\sigma` region,
        and applying the mask.

        Parameters:
            xlim,ylim (tuple or None):
                The range to display for the x- and y-axis, respectively,
                defined as a tuple of two elements ``[xmin, xmax]``. If
                the range is ``None``, the range for the xaxis range will be
                set automatically by matploltib while the yaxis limits will be
                optimised to reject extreme data points.
            show_std (bool):
                If ``True``, the :math:`n\\sigma` range above and below the
                spectrum values will be shown as a shadowed area. ``n_sigma``
                determines how many sigmas will be plotted.
            use_mask (bool):
                If ``True``, the region in which the mask is non-zero will not
                be shown in the plot.
            show_units (bool):
                If ``True``, the units will be added to the axis labels.
            n_sigma (float):
                The number of standard deviations that will be shown if
                ``show_std=True``.
            xlabel,ylabel (str or None):
                The axis labels to be passed to the plot. If not defined, the y
                axis will be labelled as ``Flux`` and the x axis as
                ``Wavelength``.
            plt_style (str):
                Matplotlib style sheet to use. Default is 'seaborn-darkgrid'.
            figure (`~matplotlib.figure.Figure` or None):
                The matplotlib `~matplotlib.figure.Figure` object from which
                the axes must be created. If ``figure=None``, a new figure will
                be created.
            return_figure (bool):
                If ``True``, the matplotlib `~matplotlib.figure.Figure` object
                used will be returned along with the axes object.

        Returns:
            axes:
                The `~matplotlib.axes.Axes` object containing the plot
                representing the spectrum. If ``return_figure=True``, a tuple
                will be returned of the form ``(ax, fig)``, where ``fig`` is
                the associated `~matplotlib.figure.Figure`.

        Example:

          >>> ax = spectrum.plot(n_sigma=3)
          >>> ax.show()

        We can change the range of the axes after the object has been created.

          >>> ax.set_xlim(6500, 6600)
          >>> ax.show()

        """

        with plt.style.context(plt_style):
            fig = plt.figure() if figure is None else figure
            ax = fig.add_subplot(111)

        if use_mask:
            value = self.masked
            wave = np.ma.array(self.wavelength.value, mask=(self.mask > 0))
        else:
            value = self.value
            wave = self.wavelength.value

        ax.plot(wave, value)

        if show_std and self.std is not None:
            if use_mask is False:
                std_masked = self.std.value
            else:
                std_masked = np.ma.array(self.std.value, mask=(self.mask > 0))
            assert n_sigma > 0, 'invalid n_sigma value.'
            ax.fill_between(wave,
                            value + n_sigma * std_masked,
                            value - n_sigma * std_masked,
                            facecolor='b', alpha=0.3)

        ax.set_xlim(xlim)

        if ylim is None:
            # Uses percentiles to get the optimal limits
            # ylim_0 = -np.percentile(-value[value < 0], 90)
            # ylim_1 = np.percentile(value[value > 0], 99.5)
            # ylim = (ylim_0, ylim_1)
            ylim = (0, None)

        ax.set_ylim(ylim)

        if xlabel is not None:
            if show_units and isinstance(self.wavelength, Quantity):
                xlabel = '$\\mathrm{{{}}}\\,[${}$]$'.format(
                    xlabel, self.wavelength.unit.to_string('latex'))
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            if show_units:
                ylabel = '$\\mathrm{{{}}}\\,[${}$]$'.format(ylabel,
                                                            self.unit.to_string('latex_inline'))
            ax.set_ylabel(ylabel)

        if return_figure:
            return (ax, fig)
        else:
            return ax
