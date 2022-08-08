# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-08-01 17:50:48
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-08-08 14:01:33

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest
from marvin import config
from marvin.tools.image import Image
from marvin.utils.general import check_versions


IMCOORDS = np.array([[275.38201798, 275.38201798],
                     [275.38201798, 281.0],
                     [275.38201798, 286.61798202],
                     [281.0, 275.38201798],
                     [281.0, 281.0],
                     [281.0, 286.61798202],
                     [286.61798202, 275.38201798],
                     [286.61798202, 281.0],
                     [286.61798202, 286.61798202]])

@pytest.fixture(scope='session')
def image():
    im = Image('8485-1901', release="DR17", mode='local')
    yield im
    im = None


class TestImage(object):

    def test_getimage(self, image):
        assert isinstance(image, Image)
        assert image.data is not None
        assert image.plateifu == '8485-1901'

    @pytest.mark.parametrize('plateifu, origin, dir3d',
                             [('8485-1901', 'file', 'stack'),
                              ('8116-1901', 'api', 'mastar')])
    def test_origins(self, plateifu, origin, dir3d):
        im = Image(plateifu)
        assert im.plateifu == plateifu
        assert im.data_origin == origin

        if not check_versions(im._drpver, 'v2_4_3'):
            assert dir3d in im._getFullPath()

    def test_release(self, plateifu):
        im = Image(plateifu, release='DR17')
        assert im.release == 'DR17'

    @pytest.mark.parametrize('plateifu, release',
                             [('8485-1901', 'DR17')])
    def test_attributes(self, plateifu, release):
        im = Image(plateifu, release=release)
        assert im.wcs is not None
        assert im.header is not None
        assert im.ra is not None
        assert hasattr(im, 'bundle')

    @pytest.mark.parametrize('cubecoord, imcoord',
                             [((17, 17), (281, 281)),
                              ((18, 18), (286.61798202, 286.61798202)),
                              ((9.99826865, 6.34193863), (241.66439915, 221.12320284))],
                             ids=['center', 'off', 'hexedge'])
    @pytest.mark.usefixtures('checkdb')
    def test_wcs(self, image, cubecoord, imcoord):
        cube = image.getCube()
        wcs = cube.wcs.celestial
        cube_radec = wcs.all_pix2world([cubecoord], 0)
        im_radec = image.wcs.all_pix2world([imcoord], 1)
        assert im_radec == pytest.approx(cube_radec, rel=1e-6)

    @pytest.mark.usefixtures('checkdb')
    def test_wcs_aperture(self, image):
        cube = image.getCube()
        wcs = cube.wcs.celestial
        aper = cube.getAperture((17, 17), 1)
        coords = list(zip(*np.where(aper.mask)))
        im_radec = image.wcs.all_pix2world(IMCOORDS, 1)
        cube_radec = wcs.all_pix2world(coords, 0)
        assert im_radec == pytest.approx(cube_radec, rel=1e-6)

    def test_saveimage(self, image, temp_scratch):
        file = temp_scratch / 'test_image.png'
        image.save(str(file))
        assert file.exists() is True

    def test_new_cutout(self, image):
        wcs = image.wcs
        image.get_new_cutout(100, 100, scale=0.192)
        assert wcs != image.wcs

    def test_fromlist(self):
        plateifu = ['8485-1901', '7443-12701']
        images = Image.from_list(plateifu)
        names = [im.plateifu for im in images]
        assert plateifu == names

    def test_getrandom(self):
        images = Image.get_random(2, release='DR17')
        assert len(images) == 2

    def test_byplate(self):
        images = Image.by_plate(8485, release='DR17')
        assert isinstance(images, list)
        assert images[0].plate == 8485

    def test_imversion(self):
        config.setRelease("DR15")
        im = Image('8485-1901', release='DR15', mode='local')
        fp = im._getFullPath()
        assert 'stack' in fp

        config.setRelease("DR17")
        im = Image('8485-1901', release='DR17', mode='local')
        fp = im._getFullPath()
        assert 'stack' not in fp


class TestBundle(object):

    def test_skies(self, image):
        assert image.bundle.skies is None
        image.bundle.get_sky_coordinates()
        assert image.bundle.skies is not None
        assert len(image.bundle.skies) == 2






