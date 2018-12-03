#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2017-10-30
# @Filename: spectrum.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-29 17:48:19


from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from astropy.units import Angstrom, CompositeUnit, Quantity

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
        pixmask_flag (str):
            The maskbit flag to be used to convert from mask bits to labels
            (e.g., MANGA_DRP3PIXMASK).
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
                mask=None, dtype=None, copy=True, pixmask_flag=None, **kwargs):

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

        obj.pixmask_flag = pixmask_flag

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.ivar = getattr(obj, 'ivar', None)
        self._std = getattr(obj, '_std', None)
        self.mask = getattr(obj, 'mask', None)
        self.wavelength = getattr(obj, 'wavelength', None)

        self.pixmask_flag = getattr(obj, 'pixmask_flag', None)

        self._set_unit(getattr(obj, 'unit', None))

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
    def std(self):
        """The standard deviation of the measurement."""

        return self.error

    def plot(self, xlim=None, ylim=None, show_std=True, use_mask=True,
             n_sigma=1, xlabel='Wavelength', ylabel='Flux', show_units=True,
             plt_style='seaborn-darkgrid', figure=None, return_figure=False,
             title=None, ytrim='positive', **kwargs):
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
                If ``True``, the region in which the mask is set to
                ``DONOTUSE`` will be masked out in the plot.
            show_units (bool):
                If ``True``, the units will be added to the axis labels.
            n_sigma (float):
                The number of standard deviations that will be shown if
                ``show_std=True``.
            xlabel,ylabel (str or None):
                The axis labels to be passed to the plot. If not defined, the y
                axis will be labelled as ``Flux`` and the x axis as
                ``Wavelength``.
            ytrim (str):
                The default y-limit behavior when no ylim specified. Can be "positive",
                "percentile", or "minmax". "positive" imposes a lower y bound of 0.
                "percentile" performs a 10% sigma clipping on the data to compute the bounds.
                "minmax" uses the straight min/max of the data.  Default is "positive".
            title (str):
                The title of the plot
            plt_style (str):
                Matplotlib style sheet to use. Default is 'seaborn-darkgrid'.
            figure (`~matplotlib.figure.Figure` or None):
                The matplotlib `~matplotlib.figure.Figure` object from which
                the axes must be created. If ``figure=None``, a new figure will
                be created.
            return_figure (bool):
                If ``True``, the matplotlib `~matplotlib.figure.Figure` object
                used will be returned along with the axes object.
            kwargs (dict):
                Keyword arguments to be passed to `~matplotlib.axes.Axes.plot`.

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

        if self.mask is None:
            use_mask = False

        if use_mask and 'DONOTUSE' in self.pixmask.schema.label.tolist():
            donotuse_mask = self.pixmask.get_mask('DONOTUSE') > 0
            value = self.masked
            wave = np.ma.array(self.wavelength.value, mask=donotuse_mask)
            std = self.std.value if self.std else None
        else:
            value = self.value
            wave = self.wavelength.value
            std = self.std.value if self.std else None

        ax.plot(wave, value, **kwargs)

        if show_std and self.std is not None:
            assert n_sigma > 0, 'invalid n_sigma value.'
            ax.fill_between(wave,
                            value + n_sigma * std,
                            value - n_sigma * std,
                            facecolor='b', alpha=0.3)

        ax.set_xlim(xlim)

        if ylim is None:
            if ytrim == 'positive':
                ylim = (0, None)
            elif ytrim == 'percentile':
                # Uses percentiles to get the optimal limits
                ylim_0 = -np.percentile(-value[value < 0], 90)
                ylim_1 = np.percentile(value[value > 0], 99.5)
                ylim = (ylim_0, ylim_1)
            elif ytrim == 'minmax':
                # let matplotlib decide
                ylim = (np.min(value), np.max(value))

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

        if title:
            ax.set_title(title)

        if return_figure:
            return (ax, fig)
        else:
            return ax
