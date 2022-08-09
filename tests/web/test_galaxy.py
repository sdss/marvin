# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-22 10:38:28
# @Last modified by:   Brian Cherinka
# @Last modified time: 2017-07-31 12:07:00

from __future__ import print_function, division, absolute_import
from marvin.web.controllers.galaxy import make_nsa_dict
from marvin.web.controllers.galaxy import getWebMap
from marvin.tools.cube import Cube
from tests.conftest import set_the_config
import pytest


pytestmark = pytest.mark.uses_web


@pytest.fixture()
def cube(galaxy, mode):
    set_the_config(galaxy.release)
    cube = Cube(plateifu=galaxy.plateifu, mode=mode, release=galaxy.release)
    cube.exp_nsa_plotcols = galaxy.nsa_data
    return cube


@pytest.fixture()
def params(galaxy):
    return {'release': galaxy.release}


@pytest.mark.parametrize('page', [('galaxy_page', 'Galaxy:index')], ids=['galaxy'], indirect=True)
class TestGalaxyPage(object):

    def test_assert_galaxy_template_used(self, page, get_templates):
        page.load_page('get', page.url)
        assert '' == page.data
        template, context = get_templates[0]
        assert 'galaxy.html' == template.name, 'Template used should be galaxy.html'


@pytest.mark.parametrize('page', [('galaxy_page', 'initnsaplot')], ids=['initnsa'], indirect=True)
class TestNSA(object):

    #@marvin_test_if(mark='skip', cube=dict(nsa=[None]))
    def test_nsadict_correct(self, cube, page):
        nsa, cols = make_nsa_dict(cube.nsa)
        for value in cube.exp_nsa_plotcols.values():
            assert set(value.keys()).issubset(set(cols))
            page.assert_dict_contains_subset(value, nsa)
            page.assertListIn(value.keys(), cols)

    @pytest.mark.skip('these magically worked when they should not have and now they actually do not')
    def test_initnsa_method_not_allowed(self, page, params, get_templates):
        page.load_page('get', page.url, params=params)
        template, context = get_templates[0]
        assert template.name == 'errors/method_not_allowed.html'

    def test_initnsa_no_plateifu(self, page, get_templates):
        errmsg = 'Field may not be null.'
        page.load_page('post', page.url)
        template, context = get_templates[0]
        page.route_no_valid_webparams(template, context, 'plateifu', reqtype='post', errmsg=errmsg)


class TestWebMap(object):

    @pytest.mark.parametrize('parameter, channel',
                             [('emline_gflux', 'ha_6564'),
                              ('emline_gsigma', 'ha_6564'),
                              ('stellar_sigma', None)],
                             ids=['gflux', 'gsigma', 'stellarsigma'])
    def test_getmap(self, cube, parameter, channel):
        webmap, mapmsg = getWebMap(cube, parameter=parameter, channel=channel)

        assert isinstance(webmap, dict)
        assert 'values' in webmap
        assert isinstance(webmap['values'], list)
        assert parameter in mapmsg
        if 'sigma' in parameter and cube.release != 'MPL-6':
            assert 'Corrected' in mapmsg

    def test_getmap_failed(self, cube):
        webmap, mapmsg = getWebMap(cube, parameter='crap')
        assert webmap is None
        assert 'Could not get map' in mapmsg





