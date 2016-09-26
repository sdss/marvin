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
import numpy as np
from marvin.core.exceptions import MarvinMissingDependency

try:
    import matplotlib.pyplot as plt
    pyplot = True
except ImportError:
    pyplot = None


class Spectrum(np.ndarray):
    """A class representing an spectrum with extra functionality.

    Parameters:
        value (array-like):
            The 1-D array contianing the spectrum.
        value_units (str, optional):
            The units of the spectrum.
        wavelength (array-like, optional):
            The wavelength solution for ``spectrum``. Must have the same number
            of elements.
        ivar (array-like, optional):
            The inverse variance array for ``spectrum``. Must have the same
            number of elements.
        mask (array-like, optional):
            The mask array for ``spectrum``. Must have the same number of
            elements.
        wavelength_unit (str, optional):
            The units of the wavelength solution.
        is_flux (bool):
            If ``True``, creates internal properties called ``flux`` and
            ``flux_units`` pointing to ``value`` and ``value_units``,
            respectively.

    """

    def __new__(cls, input_array, units=None, wavelength=None,
                wavelength_unit=None, ivar=None, mask=None, **kwargs):

        obj = np.asarray(input_array).view(cls)

        obj.units = units
        obj.wavelength = np.array(wavelength) if wavelength is not None else wavelength
        obj.wavelength_unit = wavelength_unit
        obj.ivar = np.array(ivar) if ivar is not None else ivar
        obj.mask = np.array(mask) if mask is not None else mask

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.units = getattr(obj, 'units', None)
        self.wavelength = getattr(obj, 'wavelength', None)
        self.wavelength_unit = getattr(obj, 'wavelength_unit', None)
        self.ivar = getattr(obj, 'ivar', None)
        self.mask = getattr(obj, 'mask', None)

    def __array_wrap__(self, out_arr, context=None):

        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __repr__(self):
        """Representation for Spectrum."""

        return '<Marvin Spectrum ({0!s})'.format(self)

    def plot(self, array=None, xlim=None, ylim=None, mask_color=None,
             xlabel=None, ylabel=None, figure=None, return_figure=False, **kwargs):
        """Plots a spectrum using matplotlib.

        Returns a |axes|_ object with a representation of this spectrum.
        The returned ``axes`` object can then be showed, modified, or saved to
        a file. If running Marvin from an iPython console and
        `matplotlib.pyplot.ion()
        <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.ion>`_,
        the plot will be displayed interactivelly.

        Parameters:
            array (None or {ivar', 'mask'}):
                The array to display, defaults to the internal spectrum with
                which the object was initialised.
            xlim,ylim (tuple-like or None):
                The range to display for the x- and y-axis, respectively,
                defined as a tuple of two elements ``[xmin, xmax]``. If
                the range is ``None``, the range for the axis will be set
                automatically by matploltib. If ``Spectrum.wavelength`` is
                defined, the range in the x-axis must be defined as a
                wavelength range.
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
                <http://matplotlib.org/api/colors_api.html>`_
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
            ax (`matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_):
                The matplotlib axes object containing the plot representing
                the spectrum. If ``return_figure=True``, a tuple will be
                returned of the form ``(ax, fig)``.

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

        # Alternative if this block keeps throwing an error when matplotlib is actually installed:
        # if pyplot not in sys.modules
        if not pyplot:
            raise MarvinMissingDependency('matplotlib is not installed.')

        validSpectrum = ['ivar', 'mask']
        assert array is None or array in validSpectrum, \
            'array must be one of {0!r}'.format(validSpectrum)

        if array is None:
            data = self
        elif array == 'ivar':
            data = self.ivar
        elif array == 'mask':
            data = self.mask

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
                xlabel = 'Wavelength'
                if self.wavelength_unit == 'Angstrom':
                    xlabel += r' $[\rm\AA]$'
                elif self.wavelength_unit is not None:
                    xlabel += r' [{0}]'.format(self.wavelength_unit)
            else:
                xlabel = ''

        if ylabel is None:
            if array is None:
                ylabel = 'Flux'
                if self.units is not None:
                    if self.units.lower() == '1e-17 erg/s/cm^2/ang/spaxel':
                        ylabel += r' $[\rm 10^{-17}\,erg\,s^{-1}\,cm^{-2}\,\AA^{-1}\,spaxel^{-1}]$'
                    else:
                        ylabel += r' [{0}]'.format(self.flux_units)
            elif array == 'ivar':
                ylabel = 'Inverse variance'
            else:
                ylabel = ''

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if return_figure:
            return (ax, fig)
        else:
            return ax
