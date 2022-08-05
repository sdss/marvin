# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-07 13:54:18
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-06-27 11:19:28

from __future__ import print_function, division, absolute_import
from tests.api.conftest import ApiPage
import pytest


@pytest.mark.parametrize('page', [('api', 'CubeView:index')], ids=['cubes'], indirect=True)
class TestCubeView(object):

    def test_get_cube_success(self, page, params):
        page.load_page('get', page.url, params=params)
        data = 'this is a cube'
        page.assert_success(data)


@pytest.mark.parametrize('page', [('api', 'getCube')], ids=['getcubes'], indirect=True)
class TestGetCube(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    def test_plateifu_success(self, galaxy, page, params, reqtype):
        data = {'plateifu': galaxy.plateifu, 'mangaid': galaxy.mangaid, 'ra': galaxy.ra, 'dec': galaxy.dec,
                'redshift': galaxy.redshift}
        page.load_page(reqtype, page.url.format(name=galaxy.plateifu), params=params)
        page.assert_success(data)

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('name, missing, errmsg', [(None, 'release', 'Missing data for required field.'),
                                                       ('badname', 'name', 'String does not match expected pattern.'),
                                                       ('84', 'name', 'Shorter than minimum length 4.')],
                             ids=['norelease', 'badname', 'shortname'])
    def test_plateifu_failure(self, galaxy, page, reqtype, params, name, missing, errmsg):
        if name is None:
            page.route_no_valid_params(page.url.format(name=galaxy.plateifu), missing, reqtype=reqtype, errmsg=errmsg)
        else:
            page.route_no_valid_params(page.url.format(name=name), missing, reqtype=reqtype, params=params, errmsg=errmsg)


cmsg = 'Must be one of: flux, ivar, mask, disp, predisp, specres, specresd, prespecres, prespecresd.'

@pytest.mark.slow
@pytest.mark.parametrize('page', [('api', 'getExtension')], ids=['getcubeext'], indirect=True)
class TestCubeExtension(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('cubeext', [('flux'), ('ivar'), ('mask')])
    def test_cubeext_success(self, galaxy, page, params, reqtype, cubeext):
        params.update({'name': galaxy.plateifu, 'cube_extension': cubeext})
        data = {'extension_data': []}
        page.load_page(reqtype, page.url.format(**params), params=params)
        page.assert_success(data)

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    @pytest.mark.parametrize('name, missing, errmsg, cubeext',
                             [(None, 'release', 'Missing data for required field.', 'flux'),
                              ('badname', 'name', 'String does not match expected pattern.', 'flux'),
                              ('84', 'name', 'Shorter than minimum length 4.', 'flux'),
                              ('8485-1901', 'cube_extension', cmsg, 'stuff')],
                             ids=['norelease', 'badname', 'shortname', 'badext'])
    def test_cubeext_failure(self, galaxy, page, reqtype, params, name, missing, errmsg, cubeext):
        params.update({'name': name, 'cube_extension': cubeext})
        if name is None:
            params.update({'name': galaxy.plateifu, 'cube_extension': cubeext})
            page.route_no_valid_params(page.url.format(**params), missing, reqtype=reqtype, errmsg=errmsg)
        else:
            page.route_no_valid_params(page.url.format(**params), missing, reqtype=reqtype, params=params, errmsg=errmsg)

