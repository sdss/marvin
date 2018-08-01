# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-08-01 17:50:48
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-08-01 18:29:39

from __future__ import print_function, division, absolute_import

import pytest

from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.tools.image import Image


@pytest.fixture(scope='session')
def image():
    im = Image('8485-1901')
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
        assert dir3d in im._getFullPath()

    def test_release(self, plateifu):
        im = Image(plateifu, release='MPL-5')
        assert im.release == 'MPL-5'

    @pytest.mark.parametrize('plateifu, release',
                             [('8485-1901', 'MPL-6'),
                              ('8116-1901', 'MPL-6'),
                              ('8116-1901', 'MPL-4')])
    def test_attributes(self, plateifu, release):
        im = Image(plateifu, release=release)
        if release == 'MPL-4':
            assert im.wcs is None
            assert im.header is None
            assert im.ra is None
            assert not hasattr(im, 'bundle')
        else:
            assert im.wcs is not None
            assert im.header is not None
            assert im.ra is not None
            assert hasattr(im, 'bundle')

    def test_saveimage(self, image, temp_scratch):
        file = temp_scratch.join('test_image.png')
        image.save(str(file))
        assert file.check() is True

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
        images = Image.get_random(2)
        assert len(images) == 2

    def test_byplate(self):
        images = Image.by_plate(8485)
        assert isinstance(images, list)
        assert images[0].plate == 8485


class TestBundle(object):

    def test_skies(self, image):
        assert image.bundle.skies is None
        image.bundle.get_sky_coordinates()
        assert image.bundle.skies is not None
        assert len(image.bundle.skies) == 2






