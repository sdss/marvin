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
