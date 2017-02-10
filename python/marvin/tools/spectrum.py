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
import sys
import numpy as np
import matplotlib.pyplot as plt

if 'seaborn' in sys.modules:
    import seaborn as sns
else:
    plt.style.use('seaborn-darkgrid')


class Spectrum(object):
    """A class representing an spectrum with extra functionality.

    Parameters:
        flux (array-like):
            The 1-D array contianing the spectrum.
        units (str, optional):
            The units of the flux spectrum.
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

    """

    def __init__(self, flux, units=None, wavelength_unit=None,
                 ivar=None, mask=None, wavelength=None):

        self.flux = np.array(flux)
        self.ivar = np.array(ivar) if ivar is not None else None
        self.mask = np.array(mask) if mask is not None else None
        self.wavelength = np.array(wavelength) if wavelength is not None else None

        self.units = units
        self.wavelength_unit = wavelength_unit

        # Performs some checks.

        assert len(self.flux.shape) == 1, 'spectrum must be 1-D'

        if self.ivar is not None:
            assert len(self.ivar.shape) == 1, 'ivar must be 1-D'
            assert len(self.flux) == len(self.ivar), \
                'ivar must have the same lenght as the base spectrum'

        if self.mask is not None:
            assert len(self.mask.shape) == 1, 'mask must be 1-D'
            assert len(self.flux) == len(self.mask), \
                'mask must have the same lenght as the base spectrum'

        if self.wavelength is not None:
            assert len(self.wavelength.shape) == 1, 'wavelength must be 1-D'
            assert len(self.flux) == len(self.wavelength), \
                'wavelength must have the same lenght as the base spectrum'

    def __repr__(self):
        """Representation for Spectrum."""

        return '<Marvin Spectrum ({0!s})'.format(self.flux)

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
            data = self.flux
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
            if array == 'flux':
                ylabel = 'Flux'
                if self.units == '1e-17 erg/s/cm^2/Ang/spaxel':
                    ylabel += r' $[\rm 10^{-17}\,erg\,s^{-1}\,cm^{-2}\,\AA^{-1}\,spaxel^{-1}]$'
                elif self.units is not None:
                    ylabel += r' [{0}]'.format(self.units)
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
