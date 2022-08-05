# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-06-20 16:36:37
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-11-15 17:43:10

from __future__ import print_function, division, absolute_import

import pytest


from marvin.utils.general.images import (getImagesByList, getImagesByPlate, getRandomImages,
                                         getDir3d, showImage, get_images_by_list, get_images_by_plate,
                                         get_random_images, show_image)
from marvin.tools.image import Image
from marvin.core.exceptions import MarvinDeprecationWarning, MarvinError, MarvinUserWarning


def test_getdir3d():
    exp = getDir3d('8485-1901', release='DR17')
    assert exp == 'stack'


def test_get_images_by_plate():
    ims = get_images_by_plate('8485', release='DR17')
    assert isinstance(ims[0], Image)
    assert len(ims) == 17


def test_get_images_by_list():
    ims = get_images_by_list(['8485-1901', '7443-12701'])
    assert isinstance(ims[0], Image)
    assert ims[0].plateifu == '8485-1901'
    assert ims[1].mode == 'local'
    assert ims[1].data_origin == 'file'


def test_get_random_images():
    ims = get_random_images(num=1, download=False, release='DR17')
    assert isinstance(ims[0], Image)
    assert len(ims) == 1
    assert ims[0].mode == 'remote'


def test_ShowImage_warning():
    with pytest.warns(MarvinDeprecationWarning, match=('showImage is deprecated as of Marvin 2.3.0.'
                                                       ' Please use marvin.tools.image.Image instead.')):
        showImage(plateifu='8485-1901', mode='local', release='DR17', return_image=False, show_image=False)


def test_GetImagesList_warning():
    with pytest.warns(MarvinDeprecationWarning, match=('getImagesByList is deprecated as of Marvin 2.3.0.'
                                                       ' Please use get_images_by_list')):
        ims = getImagesByList(['8485-1901'], mode='local', as_url=True, download=False, release='DR17')
        assert ims[0] == 'https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8485/images/1901.png'


def test_GetImagesPlate_warning():
    with pytest.warns(MarvinDeprecationWarning, match=('getImagesByPlate is deprecated as of Marvin 2.3.0.'
                                                       ' Please use get_images_by_plate')):
        ims = getImagesByPlate(8485, mode='local', download=False, release="DR17", as_url=True)
        assert ims[0] == 'https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8485/images/1901.png'


def test_GetRandomImages_warning():
    with pytest.warns(MarvinDeprecationWarning, match=('getRandomImages is deprecated as of Marvin 2.3.0.'
                                                       ' Please use get_random_images')):
        ims = getRandomImages(num=1, mode='local', as_url=True, release='DR17', download=False)
        assert len(ims) == 1
