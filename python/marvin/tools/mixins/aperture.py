#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-14
# @Filename: aperture.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-27 13:02:19

import astropy.coordinates
import astropy.units
import numpy


try:
    import photutils
except ImportError:
    photutils = None


__all__ = ['GetApertureMixIn', 'MarvinAperture']


class MarvinAperture(photutils.Aperture if photutils else object):
    """Extends `photutils.Aperture` allowing to extract spaxels in the aperture.

    This class is not intended for general use and it is dynamically set as
    the base for a `~photutils.Aperture` instance.

    """

    @property
    def parent(self):
        """Returns or sets the parent object."""

        if not hasattr(self, '_parent'):
            return None

        return self._parent

    @parent.setter
    def parent(self, value):

        self._parent = value

    @property
    def mask(self):
        """Returns the fractional overlap mask.

        Equivalent to using `photutils.PixelAperture.to_mask` followed
        by `photutils.ApertureMask.to_image` using the shape of the parent
        object. Combines all the apertures in a single mask.

        """

        assert self.parent is not None, 'no parent set'

        if isinstance(self, photutils.SkyAperture):
            aperture = self.to_pixel(self.parent.wcs)
        else:
            aperture = self

        mask = numpy.zeros(self.parent._shape)

        for ap in aperture.to_mask(method='exact'):
            mask += ap.to_image(shape=(self.parent._shape))

        return mask

    def getSpaxels(self, threshold=0.5, lazy=True, mask=None, **kwargs):
        """Returns the spaxels that fall within the aperture.

        Parameters
        ----------
        threshold : float
            The minimum fractional overlap between the spaxel and the aperture
            grid for the spaxel to be returned.
        lazy : bool
            Whether the returned spaxels must be fully loaded or lazily
            instantiated.
        mask : numpy.ndarray
            A mask that defines the fractional pixel overlap with the
            apertures. If ``None``, the mask returned by `.MarvinAperture.mask`
            will be used.
        kwargs : dict
            Additional parameters to be passed to the parent ``getSpaxel``
            method. Can be used to define what information is loaded
            in the spaxels.

        """

        assert threshold > 0 and threshold <= 1, 'invalid threshold value'

        if mask is None:
            mask = self.mask
        else:
            assert mask.shape == self.parent._shape, 'invalid mask shape'

        spaxel_coords = numpy.where(mask >= threshold)

        if len(spaxel_coords[0]) == 0:
            return []

        return self.parent.getSpaxel(x=spaxel_coords[1], y=spaxel_coords[0],
                                     xyorig='lower', lazy=lazy, **kwargs)


class GetApertureMixIn(object):

    def getAperture(self, coords, aperture_params, aperture_type='circular',
                    coord_type='pixel'):
        """Defines an aperture.

        This method is mostly a wrapper around the aperture classes defined in
        `photutils <http://photutils.readthedocs.io>`_. It allows to

        Parameters
        ----------
        coords : tuple or `~numpy.ndarray`
            Either a 2-element tuple ``(x, y)`` or ``(ra, dec)`` to define the
            centre of a single aperture, or a list of N tuples or a Nx2 array
            to define N apertures.
        aperture_params : tuple
            A tuple with the parameters of the aperture.

            * For ``aperture_type='rectangular'``:

              * If ``coord_type='pixel'``, a tuple ``(w, h, theta)`` where
                ``w`` is the full width of the aperture (for ``theta=0`` the
                width side is along the ``x`` axis); ``h`` is the full height
                of the aperture (for ``theta=0`` the height side is along the
                ``y`` axis); and ``theta`` is the rotation angle in radians of
                the width (``w``) side from the positive ``x`` axis (the
                rotation angle increases counterclockwise).
              * If ``coord_type='sky'``, same format but ``w`` and ``h`` must
                be in arcsec and ``theta`` is the position angle (in degrees)
                of the width side. The position angle increases
                counterclockwise, from North (PA=0) towards East.

            * For ``aperture_type='circular'``:

              * The radius ``r`` in units of pixels or arcsec depending on the
                value of ``coord_type``. Can be a tuple or a float.

            * For ``aperture_type='elliptical'``:

              * If ``coord_type='pixel'``, a tuple ``(a, b, theta)`` where
                ``a`` and ``b`` are the semi-major and semi-minor axes of the
                ellipse, respectively, and ``theta`` is the rotation angle in
                radians of the semi-major axis from the positive x axis (the
                rotation angle increases counterclockwise). If
                ``coord_type='sky'``, ``a`` and ``b`` must be in arcsec, and
                ``theta`` is the position angle (in degrees) of the semi-major
                axis. The position angle increases counterclockwise, from North
                (PA=0) towards East.

        aperture_type : {'rectangular', 'circular', 'elliptical'}
            The type of aperture to define.
        coord_type : {'pixel', 'sky'}
            Determines whether the coordinates and aperture parameters refer
            to to the frame of the image or to sky coordinates. The conversion
            between the image and sky frames is determined using the WCS
            headers from the image.

        Returns
        -------
        marvin_aperture : MarvinAperture object
            A `.MarvinAperture` instance with the definition of the aperture,
            which can be used to extract the associated spaxels or to return
            the mask.

        Examples
        --------

        A single circular aperture with a radius of 3 pixels can created as ::

            >>> cube = marvin.tools.Cube('8485-1901')
            >>> aperture = cube.getAperture((17, 15), 3)

        ``aperture`` can then be used to return all spaxels that have a
        fractional overlap with the aperture of more than 0.6 pixels ::

            >>> spaxels = aperture.getSpaxels(threshold=0.6, lazy=True)
            >>> spaxels[0]
            <Marvin Spaxel (x=15, y=13, loaded=False)

        Apertures can also be defined from sky coordinates. To define two
        elliptical apertures with semi-axes 3 and 1 arcsec and rotated
        30 degrees we do ::

            >>> ap_ell = cube.getAperture([(232.546173, 48.6892288), (232.544069, 48.6906177)],
                                          (3, 1, 30), aperture_type='elliptical')
            >>> ap_ell
            <MarvinAperture([[232.546173 ,  48.6892288],
                 [232.544069 ,  48.6906177]], a=3.0, b=1.0, theta=30.0)>

        """

        if photutils is None:
            raise ImportError('this feature requires photutils. Install it by '
                              'doing pip install photutils.')

        assert coord_type in ['pixel', 'sky'], 'invalid coord_type'

        if isinstance(coords, astropy.coordinates.SkyCoord):
            coord_type = 'sky'
        else:
            coords = numpy.atleast_2d(coords)
            if coord_type == 'sky':
                coords = astropy.coordinates.SkyCoord(coords, unit='deg')

        aperture_params = numpy.atleast_1d(aperture_params).tolist()

        if aperture_type == 'circular':
            if coord_type == 'pixel':
                ApertureClass = photutils.CircularAperture
            else:
                ApertureClass = photutils.SkyCircularAperture
        elif aperture_type == 'elliptical':
            if coord_type == 'pixel':
                ApertureClass = photutils.EllipticalAperture
            else:
                ApertureClass = photutils.SkyEllipticalAperture
        elif aperture_type == 'rectangular':
            if coord_type == 'pixel':
                ApertureClass = photutils.RectangularAperture
            else:
                ApertureClass = photutils.SkyRectangularAperture
        else:
            raise ValueError('invalid aperture_type')

        # If on-sky, converts aperture parameters to quantities
        if coord_type == 'sky':

            if aperture_type == 'circular':
                n_params = 1
            else:
                n_params = 3

            assert len(aperture_params) == n_params, 'invalid number of parameters'

            units = [astropy.units.arcsec, astropy.units.arcsec, astropy.units.deg]

            for ii in range(n_params):
                if not isinstance(aperture_params[ii], astropy.units.Quantity):
                    aperture_params[ii] *= units[ii]

        aperture = ApertureClass(coords, *aperture_params)

        # Overrides the aperture class so that it inherits from MarvinAperture and
        # can gain the methods we defined there. Sets the parent to self.
        aperture.__class__ = type('MarvinAperture', (ApertureClass, MarvinAperture), {})
        aperture.parent = self

        return aperture
