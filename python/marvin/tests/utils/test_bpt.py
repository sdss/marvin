#!/usr/bin/env python
# encoding: utf-8
#
# test_bpt_pytest.py
#
# Created by José Sánchez-Gallego on 10 Feb 2017.
# Converted to pytest by Brett Andrews on 15 Jun 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from matplotlib import pyplot as plt
import numpy as np
import pytest

from marvin.tools.maps import Maps
<<<<<<< HEAD
from marvin.tests import UseReleases, UseBintypes


@UseBintypes('SPX')
@UseReleases('MPL-5')
class TestBPT(object):

    emission_mechanisms = ['sf', 'comp', 'agn', 'seyfert', 'liner', 'invalid', 'ambiguous']

    def _run_tests_8485_1901(self, maps):

        masks, figure = maps.get_bpt(show_plot=False, return_figure=True, use_oi=True)
        assert isinstance(figure, plt.Figure)

        for em_mech in self.emission_mechanisms:
            assert em_mech in masks.keys()

        assert np.sum(masks['sf']['global']) == 62
        assert np.sum(masks['comp']['global']) == 1

        for em_mech in ['agn', 'seyfert', 'liner']:
            assert np.sum(masks[em_mech]['global']) == 0
=======
from marvin.tests import marvin_test_if_class
from marvin.core.exceptions import MarvinDeprecationWarning
>>>>>>> f063c3e4e0fa4339104dff46e56edb5ab6a9b65b


@pytest.fixture()
def maps(galaxy, mode):
    maps = Maps(plateifu=galaxy.plateifu, mode=mode)
    maps.bptsums = galaxy.bptsums if hasattr(galaxy, 'bptsums') else None
    yield maps
    maps = None


@marvin_test_if_class(mode='skip', maps=dict(release=['MPL-4']))
class TestBPT(object):

    mechanisms = ['sf', 'comp', 'agn', 'seyfert', 'liner', 'invalid', 'ambiguous']

    @pytest.mark.parametrize('useoi', [(True), (False)], ids=['withoi', 'nooi'])
    def test_bpt(self, maps, useoi):
        if maps.bptsums is None:
            pytest.skip('no bpt data found in galaxy test data')

        bptflag = 'nooi' if useoi is False else 'global'

        masks, figure = maps.get_bpt(show_plot=False, return_figure=True, use_oi=useoi)
        assert isinstance(figure, plt.Figure)

        for mech in self.mechanisms:
            assert mech in masks.keys()
            assert np.sum(masks[mech]['global']) == maps.bptsums[bptflag][mech]

    def test_bpt_diffsn(self, maps):
        if maps.bptsums is None:
            pytest.skip('no bpt data found in galaxy test data')

        masks, figure = maps.get_bpt(show_plot=False, return_figure=True, use_oi=True, snr_min=5)
        assert isinstance(figure, plt.Figure)

        for mech in self.mechanisms:
            assert mech in masks.keys()
            assert np.sum(masks[mech]['global']) == maps.bptsums['snrmin5'][mech]

    def test_bpt_oldsn(self, maps):
        if maps.bptsums is None:
            pytest.skip('no bpt data found in galaxy test data')

        with pytest.warns(MarvinDeprecationWarning) as record:
            masks = maps.get_bpt(snr=5, return_figure=False, show_plot=False)
        assert len(record) == 1
        assert record[0].message.args[0] == "snr is deprecated. Use snr_min instead. snr will be removed in a future version of marvin"

        for mech in self.mechanisms:
            assert mech in masks.keys()
            assert np.sum(masks[mech]['global']) == maps.bptsums['snrmin5'][mech]

