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
from marvin.utils.general.images import getImagesByList, getImagesByPlate, getRandomImages, getDir3d, showImage
from marvin.utils.general import check_versions
from marvin.tests.conftest import Galaxy, tempafile
from marvin.tests import marvin_test_if
from marvin.core.exceptions import MarvinError, MarvinUserWarning
import pytest
import os
import warnings


try:
    from sdss_access import Access, AccessError
except ImportError:
    Path = None
    Access = None


imagelist = ['8485-1901', '7443-12701', '7443-1901']
newgals = ['7495-1901']


@pytest.fixture(scope='function')
def rsync(mode):
    ''' fixture to create generic rsync object '''

    rsync = Access(label='marvin_getlist', verbose=False)
    if mode != 'local':
        rsync.remote()
    yield rsync
    rsync.reset()
    rsync = None


localredux = os.getenv('MANGA_SPECTRO_REDUX')
remoteredux = 'https://sdss@dtn01.sdss.org/sas/mangawork/manga/spectro/redux'
remoteurl = 'https://data.sdss.org/sas/mangawork/manga/spectro/redux'
bases = [localredux, remoteredux, remoteurl]
rmodes = ['full', 'url']


@pytest.fixture()
def base(mode, asurl):
    if asurl is False:
        return localredux
    else:
        if mode != 'local':
            return remoteredux
        else:
            return remoteurl


@pytest.fixture(scope='session', params=newgals)
def newgalaxy(request, maindb, get_params, saslocal):
    release, bintype, template = get_params

    gal = Galaxy(request.param)
    gal.set_params(bintype=bintype, template=template)
    gal.set_filepaths()
    yield gal


@pytest.fixture()
def get_cube(newgalaxy, rsync):
    if not os.path.isfile(newgalaxy.cubepath):
        rsync.add('mangacube', **newgalaxy.access_kwargs)
        rsync.set_stream()
        rsync.commit()
    yield newgalaxy


@pytest.fixture(params=rmodes)
def asurl(request):
    if request.param == 'full':
        return False
    elif request.param == 'url':
        return True


@pytest.fixture()
def make_paths(request, rsync, mode, asurl, release):
    inputs = request.param if hasattr(request, 'param') else None
    rmode = 'url' if asurl else 'full'
    fullpaths = []
    inputs = inputs if inputs else imagelist
    for plateifu in inputs:
        gal = Galaxy(plateifu)
        gal.set_params(release=release)
        gal.set_filepaths()
        name = 'mangaimage'
        if mode == 'local':
            path = rsync.__getattribute__(rmode)(name, **gal.access_kwargs)
            fullpaths.append(path)
        else:
            rsync.add(name, **gal.access_kwargs)
            rsync.set_stream()
            path = rsync.get_urls() if asurl else rsync.get_paths()
            fullpaths.extend(path)
    return fullpaths


class TestImagesGetDir3d(object):

    @pytest.mark.parametrize('expval', [('stack')])
    def test_getdir3d(self, galaxy, expval, mode, db):
        dir3d = getDir3d(galaxy.plateifu, mode=mode, release=galaxy.release)
        assert expval == dir3d

    @pytest.mark.parametrize('expval', [('stack')])
    def test_getdir3d_plate(self, galaxy, expval, mode, db):
        dir3d = getDir3d(galaxy.plate, mode=mode, release=galaxy.release)
        assert expval == dir3d


@pytest.mark.xfail()
@pytest.mark.timeout(40)
class TestImagesByList(object):

    @pytest.mark.parametrize('imglist, mode, errmsg',
                             [('7495-1901', 'local', 'Input must be of type list or Numpy array'),
                              (['nogoodid'], 'local', 'Input must be of type plate-ifu or mangaid'),
                              (imagelist, 'notvalidmode', 'Mode must be either auto, local, or remote')],
                             ids=['notlist', 'badid', 'badmode'])
    def test_failures(self, imglist, mode, errmsg, release):
        with pytest.raises(AssertionError) as cm:
            image = getImagesByList(imglist, mode=mode, release=release)
        assert cm.type == AssertionError
        assert errmsg in str(cm.value)

    def test_get_imagelist(self, make_paths, mode, asurl, release):
        images = getImagesByList(imagelist, mode=mode, as_url=asurl, release=release)
        assert set(make_paths) == set(images)

    # @pytest.mark.parametrize('make_paths', [(['7495-1901'])], indirect=True, ids=['newplateifu'])
    # def test_download(self, monkeymanga, temp_scratch, get_cube):
    #     imgpath = tempafile(get_cube.imgpath, temp_scratch)
    #     #assert os.path.isfile(get_cube.imgpath) is False
    #     assert imgpath.check(file=0) is True
    #     image = getImagesByList([get_cube.plateifu], mode='remote', as_url=True, download=True, release=get_cube.release)
    #     #assert os.path.isfile(get_cube.imgpath) is True
    #     assert imgpath.check(file=1) is True
    #     assert image is None

    # @pytest.mark.parametrize('make_paths', [(['7495-1901'])], indirect=True, ids=['newplateifu'])
    # def test_download_fails(self, monkeymanga, temp_scratch, get_cube):
    #     imgpath = tempafile(get_cube.imgpath, temp_scratch)
    #     assert imgpath.check(file=0) is True
    #     errmsg = 'Download not available when in local mode'
    #     with warnings.catch_warnings(record=True) as cm:
    #         warnings.simplefilter('always')
    #         image = getImagesByList([get_cube.plateifu], mode='local', as_url=True, download=True)
    #     assert cm[-1].category is MarvinUserWarning
    #     assert errmsg in str(cm[-1].message)


class TestImagesByPlate(object):

    @pytest.mark.parametrize('plateid, mode, errmsg',
                             [('8485abcd', 'local', 'Plateid must be a numeric integer value'),
                              (None, 'notvalidmode', 'Mode must be either auto, local, or remote')],
                             ids=['badid', 'badmode'])
    def test_failures(self, galaxy, plateid, mode, errmsg):
        plateid = plateid if plateid else galaxy.plate
        with pytest.raises(AssertionError) as cm:
            image = getImagesByPlate(plateid, mode=mode, release=galaxy.release)
        assert cm.type == AssertionError
        assert errmsg in str(cm.value)

    @pytest.mark.parametrize('make_paths, plate', [(['8485-1901'], '8485')], indirect=['make_paths'], ids=['plateifu'])
    def test_get_imageplate(self, make_paths, plate, mode, asurl, release):
        images = getImagesByPlate(plate, mode=mode, as_url=asurl, release=release)
        assert make_paths[0] in images

    # @pytest.mark.parametrize('make_paths', [(['7495-1901'])], indirect=True, ids=['newplateifu'])
    # def test_download(self, monkeymanga, temp_scratch, get_cube):
    #     imgpath = tempafile(get_cube.imgpath, temp_scratch)
    #     assert imgpath.check(file=0) is True
    #     image = getImagesByPlate(get_cube.plate, mode='remote', as_url=True, download=True)
    #     assert imgpath.check(file=1) is True
    #     assert image is None

    # def test_get_images_download_local_fail(self, monkeymanga, temp_scratch, get_cube):
    #     imgpath = tempafile(get_cube.imgpath, temp_scratch)
    #     assert imgpath.check(file=0) is True
    #     errmsg = 'Download not available when in local mode'
    #     with warnings.catch_warnings(record=True) as cm:
    #         warnings.simplefilter("always")
    #         image = getImagesByPlate(self.new_plate, mode='local', as_url=True, download=True)
    #     self.assertIs(cm[-1].category, MarvinUserWarning)
    #     self.assertIn(errmsg, str(cm[-1].message))


@pytest.mark.xfail()
class TestRandomImages(object):

    @pytest.mark.parametrize('mode, errmsg',
                             [('notvalidmode', 'Mode must be either auto, local, or remote')],
                             ids=['badmode'])
    def test_failures(self, mode, errmsg, release):
        with pytest.raises(AssertionError) as cm:
            image = getRandomImages(mode=mode, release=release)
        assert cm.type == AssertionError
        assert errmsg in str(cm.value)

    @pytest.mark.parametrize('num', [(10), (5)], ids=['num10', 'num5'])
    def test_get_image_random(self, base, num, mode, asurl, release):
        images = getRandomImages(num=num, mode=mode, as_url=asurl, release=release)
        assert images is not None
        assert num == len(images)
        assert isinstance(images, list) is True
        assert base in images[0]


class TestShowImage(object):

    def _assert_image(self, galaxy, image):
        assert image is not None
        assert image.size == (562, 562)
        assert image.format == 'PNG'
        assert str(galaxy.plate) in image.filename
        assert galaxy.ifu in image.filename

    @pytest.mark.parametrize('return_image', [(True), (False)], ids=['returnyes', 'returnno'])
    def test_show_image(self, galaxy, mode, return_image):
        image = showImage(plateifu=galaxy.plateifu, mode=mode, release=galaxy.release, return_image=return_image, show_image=False)
        if return_image:
            self._assert_image(galaxy, image)
        else:
            assert image is None

        return image

    @pytest.mark.parametrize('param, error, errmsg',
                             [({'mode': 'notvalidmode'}, AssertionError, 'Mode must be either auto, local, or remote'),
                              ({}, AssertionError, 'A filepath or plateifu must be specified!'),
                              ({'plateifu': '8485-1905'}, MarvinError, 'Error: remote filepath'),
                              ({'path': '/tmp/image.png'}, MarvinError, 'Error: local filepath /tmp/image.png does not exist.'),
                              ({'path': ['/tmp/image.png', '/tmp/image1.png']}, MarvinError, 'showImage currently only works on a single input at a time')],
                             ids=['badmode', 'noinput', 'badplateifu', 'badfile', 'toomany'])
    def test_failures(self, param, error, errmsg, release):
        if 'mode' not in param:
            param.update({'mode': 'auto'})

        with pytest.raises(error) as cm:
            image = showImage(release=release, **param)
        assert cm.type == error
        assert errmsg in str(cm.value)

    def test_withpaths(self, galaxy, mode):
        if mode != 'local':
            galaxy.set_params()
            galaxy.set_filepaths(pathtype='url')
        image = showImage(path=galaxy.imgpath, mode=mode, return_image=True, show_image=False)
        self._assert_image(galaxy, image)

    @pytest.mark.parametrize('base, mode, errmsg',
                             [(localredux, 'remote', 'Local path not allowed in remote mode'),
                              (remoteurl, 'local', 'Remote url path not allowed in local mode')],
                             ids=['remoteuserdir', 'localhttp'])
    def test_path_fails_wrongmodes(self, base, galaxy, mode, errmsg):
        path = os.path.join(base, galaxy.get_location(galaxy.imgpath))
        with pytest.raises(MarvinError) as cm:
            image = showImage(release=galaxy.release, path=path, mode=mode)
        assert cm.type == MarvinError
        assert errmsg in str(cm.value)
