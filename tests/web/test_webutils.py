# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-07-12 15:18:38
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-12 15:31:26

from __future__ import print_function, division, absolute_import
from marvin.web.web_utils import buildImageDict
from marvin.utils.general.images import getImagesByPlate


class TestBuildImageDict(object):

    def test_placeholder(self):
        images = buildImageDict(None, test=True, num=5)
        assert len(images) == 5
        assert images[0]['name'] == '4444-0000'
        assert 'placehold.it' in images[0]['image']

    def test_noimages(self):
        images = buildImageDict(None, num=5)
        assert not images

    def test_realimages(self, release):
        imfiles = getImagesByPlate(plateid='8485', as_url=True, mode='local', release=release)
        images = buildImageDict(imfiles)
        names = [im['name'] for im in images]
        assert images
        assert '8485-1901' in names
        assert 'https://data.sdss.org/sas/dr17/manga/spectro/redux/' in images[0]['image']

