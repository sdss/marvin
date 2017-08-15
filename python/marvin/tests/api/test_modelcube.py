# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-05-19 17:00:49
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-07-30 19:23:16

from __future__ import print_function, division, absolute_import
from marvin.tests.api.conftest import ApiPage
from marvin import config
import pytest


@pytest.mark.parametrize('page', [('api', 'getModelCube')], ids=['getModelCube'], indirect=True)
class TestGetModelCubes(object):

    @pytest.mark.parametrize('reqtype', [('get'), ('post')])
    def test_modelcube_success(self, galaxy, page, params, reqtype):
        if galaxy.release == 'MPL-4':
            pytest.skip('MPL-4 does not have modelcubes')
        params.update({'name': galaxy.plateifu, 'bintype': galaxy.bintype.name,
                       'template': galaxy.template.name})
        data = {'plateifu': galaxy.plateifu, 'mangaid': galaxy.mangaid,
                'bintype': galaxy.bintype.name, 'template': galaxy.template.name,
                'shape': galaxy.shape, 'redcorr': []}
        page.load_page(reqtype, page.url.format(**params), params=params)
        page.assert_success(data)

    @pytest.mark.parametrize('name, missing, errmsg, bintype, template',
                             [(None, 'release', 'Missing data for required field.', None, None),
                              ('badname', 'name', 'String does not match expected pattern.', None, None),
                              ('84', 'name', 'Shorter than minimum length 4.', None, None),
                              ('8485-1901', 'bintype', 'Not a valid choice.', 'SPVOR', 'GAU-MILESHC'),
                              ('8485-1901', 'template', 'Not a valid choice.', 'SPX', 'MILESHC'),
                              ('8485-1901', 'bintype', 'Not a valid choice.', 'STONY', 'GAU-MILESHC'),
                              ('8485-1901', 'template', 'Not a valid choice.', 'SPX', 'MILES'),
                              ('8485-1901', 'bintype', 'Field may not be null.', None, None),
                              ('8485-1901', 'template', 'Field may not be null.', 'SPX', None)],
                             ids=['norelease', 'badname', 'shortname', 'badbintype', 'badtemplate',
                                  'wrongmplbintype', 'wrongmpltemplate', 'nobintype', 'notemplate'])
    def test_modelcube_failures(self, galaxy, page, params, name, missing, errmsg, bintype, template):
        params.update({'name': name, 'bintype': bintype, 'template': template})
        if name is None:
            page.route_no_valid_params(page.url, missing, reqtype='post', errmsg=errmsg)
        else:
            url = page.url.format(**params)
            url = url.replace('None/', '') if missing in ['bintype', 'template'] else url
            page.route_no_valid_params(url, missing, reqtype='post', params=params, errmsg=errmsg)
