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
from marvin.tools.core.exceptions import MarvinMissingDependence

try:
    import matplotlib as mpl
except:
    mpl = None


class Spectrum(np.ndarray):
    """A class representing an spectrum with extra functionality.

    This class inherits from |numpy_array|_.

    Parameters:
        spectrum (array-like):
            The 1-D array contianing the spectrum.
        ivar (array-like, optional):
            The inverse variance array for ``spectrum``. Must have the same
            number of elements.
        mask (array-like, optional):
            The mask array for ``spectrum``. Must have the same number of
            elements.
        wavelength (array-like, optional):
            The wavelength solution for ``spectrum``. Must have the same number
            of elements.

    """

    def __new__(cls, *args, **kwargs):
        """Creates a new np.ndarray.

        See http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html

        """

        # We create an empty array of size 1 that we'll later resize to the
        # correct size of the array. This is so that Spectrum can be
        # subclassed. The .copy() allows to circunvent some problems when
        # trying to resize the array. There may be a better way of doing this.
        obj = np.asarray([0.0]).view(cls).copy()

        return obj

    def __init__(self, spectrum, ivar=None, mask=None, wavelength=None):

        self.ivar = np.array(ivar) if ivar is not None else None
        self.mask = np.array(mask) if mask is not None else None
        self.wavelength = np.array(wavelength) \
            if wavelength is not None else None

        # Resizes the array to the size of the input spectrum.
        # refcheck circunvents a problem with:
        # "cannot resize an array that references or is referenced
        # by another array in this way."
        self.resize(len(spectrum), refcheck=False)
        self[:] = np.array(spectrum)

        # Performs some checks.

        assert len(self.shape) == 1, 'spectrum must be 1-D'

        if self.ivar is not None:
            assert len(self.ivar.shape) == 1, 'ivar must be 1-D'
            assert len(self) == len(self.ivar), \
                'ivar must have the same lenght as the base spectrum'

        if self.mask is not None:
            assert len(self.mask.shape) == 1, 'mask must be 1-D'
            assert len(self) == len(self.mask), \
                'mask must have the same lenght as the base spectrum'

        if self.wavelength is not None:
            assert len(self.wavelength.shape) == 1, 'wavelength must be 1-D'
            assert len(self) == len(self.wavelength), \
                'wavelength must have the same lenght as the base spectrum'

    def __repr__(self):
        """Representation for Spectrum."""

        return '<Marvin Spectrum ({0!s})'.format(self)

    def plot(self, array='spectrum', xlim=None, ylim=None, mask_color=None, figure=None,
             return_figure=False, **kwargs):
        """Plots and spectrum using matplotlib.

        Returns a |axes|_ object with a representation of this spectrum.
        The returned ``axes`` object can then be showed, modified, or saved to
        a file. If running Marvin from an iPython console and
        `matplotlib.pyplot.ion()
        <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.ion>`_,
        the plot will be displayed interactivelly.

        Parameters:
            array ({'spectrum', 'ivar', 'mask'}):
                The array to display, defaults to the internal spectrum with
                which the object was initialised.
            xlim,ylim (tuple-like or None):
                The range to display for the x- and y-axis, respectively,
                defined as a tuple of two elements ``[xmin, xmax]``. If
                the range is ``None``, the range for the axis will be set
                automatically by matploltib. If ``Spectrum.wavelength`` is
                defined, the range in the x-axis must be defined as a
                wavelength range.
            mask_color (matplotlib valid color or None):
                If set and ``Spectrum.mask`` is defined, the elements of
                ``array`` with ``mask>0`` will be coloured using that value.
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

        if not mpl:
            raise MarvinMissingDependence('matplotlib is not installed.')

        array = array.lower()
        validSpectrum = ['spectrum', 'flux', 'ivar', 'mask']
        assert array in validSpectrum, 'array must be one of {0!r}'.format(validSpectrum)

        if array in ['spectrum', 'flux']:
            data = self
        elif array == 'ivar':
            data = self.ivar
        elif array == 'mask':
            data = self.mask

        xaxis = self.wavelength if self.wavelength is not None else np.arange(len(self))

        fig = mpl.pyplot.figure() if figure is None else figure
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

        if return_figure:
            return (ax, fig)
        else:
            return ax
