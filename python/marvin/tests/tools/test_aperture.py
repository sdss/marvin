#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-16
# @Filename: test_aperture.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-27 13:06:25

import numpy
import pytest

import marvin


# Inputs are [class, plateifu, release, coords,
#             aperture_params, aperture_type, coord_type, threshold]
# Outputs are [n_pixels, n_spaxels]. n_pixels are pixels on the mask that have
# a value of 1. n_spaxels are the number of spaxels extracted given the threshold.
test_params = [

    [['Cube', '8485-1901', 'MPL-6', (17, 17), 3, 'circular', 'pixel', 0.5], [21, 25]],
    [['Maps', '8485-1901', 'MPL-6', (17, 17), 3, 'circular', 'pixel', 0.5], [21, 25]],
    [['ModelCube', '8485-1901', 'MPL-6', (17, 17), 3, 'circular', 'pixel', 0.5], [21, 25]],

    [['Maps', '8485-1901', 'MPL-6', (232.5447, 48.690201), 1, 'circular', 'sky', 0.8], [5, 9]],

    [['Maps', '8485-1901', 'MPL-6', [(5, 5), (20, 20)], (4, 4, 0), 'rectangular', 'pixel', 0.5],
     [18, 42]],

    [['Maps', '8485-1901', 'MPL-6', (5, 5), (4, 4, 45), 'elliptical', 'pixel', 0.5], [9, 45]],

]


@pytest.mark.parametrize('inputs, outputs', test_params)
def test_get_aperture(inputs, outputs):

    class_name, plateifu, release, coords, \
        aperture_params, aperture_type, coord_type, threshold = inputs

    aperture_params = numpy.atleast_1d(aperture_params)

    n_pixels_mask, n_spaxels = outputs

    ToolClass = getattr(marvin.tools, class_name)
    instance = ToolClass(plateifu, release=release)

    assert isinstance(instance, ToolClass)
    assert hasattr(instance, 'getAperture')

    aperture = instance.getAperture(coords, aperture_params, aperture_type=aperture_type,
                                    coord_type=coord_type)

    assert isinstance(aperture, marvin.tools.mixins.MarvinAperture)

    if coord_type == 'pixel':
        aperture_positions = aperture.positions
    else:
        aperture_positions = numpy.array([aperture.positions.ra.deg,
                                          aperture.positions.dec.deg]).T

    numpy.testing.assert_allclose(numpy.atleast_2d(coords), aperture_positions)

    if aperture_type == 'circular':
        if coord_type == 'pixel':
            assert aperture.r == pytest.approx(aperture_params[0], rel=1e-6)
        else:
            assert aperture.r.value == pytest.approx(aperture_params[0], rel=1e-6)
    elif aperture_type == 'rectangular':
        if coord_type == 'pixel':
            assert aperture.w == pytest.approx(aperture_params[0], rel=1e-6)
            assert aperture.h == pytest.approx(aperture_params[1], rel=1e-6)
            assert aperture.theta == pytest.approx(aperture_params[2], rel=1e-6)
        else:
            assert aperture.w.value == pytest.approx(aperture_params[0], rel=1e-6)
            assert aperture.h.value == pytest.approx(aperture_params[1], rel=1e-6)
            assert aperture.theta.value == pytest.approx(aperture_params[2], rel=1e-6)
    elif aperture_type == 'elliptical':
        if coord_type == 'pixel':
            assert aperture.a == pytest.approx(aperture_params[0], rel=1e-6)
            assert aperture.b == pytest.approx(aperture_params[1], rel=1e-6)
            assert aperture.theta == pytest.approx(aperture_params[2], rel=1e-6)
        else:
            assert aperture.a.value == pytest.approx(aperture_params[0], rel=1e-6)
            assert aperture.b.value == pytest.approx(aperture_params[1], rel=1e-6)
            assert aperture.theta.value == pytest.approx(aperture_params[2], rel=1e-6)

    assert numpy.sum(aperture.mask == 1) == n_pixels_mask

    spaxels = aperture.getSpaxels(threshold=threshold)
    assert len(spaxels) == n_spaxels

    assert spaxels[0].loaded is False

    # Check that we can load the spaxel
    spaxels[0].load()
    assert spaxels[0].loaded is True


def test_get_aperture_no_lazy():
    """Tests if the lazy=False feature in getSpaxels works."""

    cube = marvin.tools.Cube('8485-1901')

    aperture = cube.getAperture((17, 17), 1)

    assert aperture.getSpaxels(lazy=False)[0].loaded is True


def test_get_spaxels_custom_mask():
    """Tests the mask parameter in getSpaxels."""

    cube = marvin.tools.Cube('8485-1901')

    aperture = cube.getAperture((17, 17), 2)

    my_mask = aperture.mask.copy()
    my_mask[0:5, 0:5] = 1

    assert len(aperture.getSpaxels(mask=my_mask)) == 34
