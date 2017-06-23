# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-02-22 10:38:28
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-05-07 14:27:07

from __future__ import print_function, division, absolute_import
from marvin.web.controllers.galaxy import make_nsa_dict
from marvin.tools.cube import Cube
from marvin import config
import pytest
from marvin.tests.web.conftest import Page


@pytest.fixture()
def cube(get_plateifu):
    return Cube(plateifu=get_plateifu, mode=config.mode)


@pytest.fixture()
def page(client, request, init_web):
    blue, endpoint = request.param
    page = Page(client, blue, endpoint)
    yield page


@pytest.fixture()
def params(release):
    return {'release': release}


@pytest.mark.parametrize('page', [('galaxy_page', 'Galaxy:index')], ids=['galaxy'], indirect=True)
class TestGalaxyPage(object):

    def test_assert_galaxy_template_used(self, page, get_templates):
        page.load_page('get', page.url)
        assert '' == page.data
        template, context = get_templates[0]
        assert 'galaxy.html' == template.name, 'Template used should be galaxy.html'


@pytest.mark.parametrize('page', [('galaxy_page', 'initnsaplot')], ids=['initnsa'], indirect=True)
class TestNSA(object):

    @pytest.mark.parametrize('exp_nsa_plotcols', [{'elpetro_absmag_i': -19.1125469207764,
                                                   'elpetro_mag_g_r': 0.64608402745868077, 'z': 0.0407447,
                                                   'elpetro_th50_r': 1.33067, 'elpetro_logmass': 9.565475912843823,
                                                   'elpetro_ba': 0.87454, 'elpetro_mag_i_z': 0.2311751372102151,
                                                   'elpetro_phi': 154.873, 'elpetro_mtol_i': 1.30610692501068,
                                                   'elpetro_th90_r': 3.6882, 'elpetro_mag_u_r': 1.8892372699482216,
                                                   'sersic_n': 3.29617}], ids=['nsaplotcols'])
    def test_nsadict_correct(self, page, cube, exp_nsa_plotcols):
        nsa, cols = make_nsa_dict(cube.nsa)
        assert exp_nsa_plotcols.items() <= nsa.items()
        page.assert_dict_contains_subset(exp_nsa_plotcols, nsa)
        page.assertListIn(exp_nsa_plotcols.keys(), cols)

    def test_initnsa_method_not_allowed(self, page, params, get_templates):
        page.load_page('get', page.url, params=params)
        template, context = get_templates[0]
        assert template.name == 'errors/method_not_allowed.html'

    def test_initnsa_no_plateifu(self, page, get_templates):
        errmsg = 'Field may not be null.'
        page.load_page('post', page.url)
        template, context = get_templates[0]
        page.route_no_valid_webparams(template, context, 'plateifu', reqtype='post', errmsg=errmsg)


