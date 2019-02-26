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

import inspect

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import LocatableAxes
import numpy as np
import pytest

from marvin.core.exceptions import MarvinError
from marvin.tools.maps import Maps
from marvin.tests import marvin_test_if_class
from marvin.utils.dap.bpt import get_snr, bpt_kewley06
from marvin.core.exceptions import MarvinDeprecationWarning


@pytest.fixture()
def maps(galaxy, mode):
    if galaxy.bintype.name != 'SPX':
        pytest.skip('Only running one bintype for bpt tests')

    # if galaxy.release != 'MPL-6':
    #     pytest.skip('Explicitly skipping here since marvin_test_if_class does not work in 2.7')

    maps = Maps(plateifu=galaxy.plateifu, mode=mode)
    maps.bptsums = galaxy.bptsums if hasattr(galaxy, 'bptsums') else None
    yield maps
    maps = None


@marvin_test_if_class(mark='skip', maps=dict(release=['MPL-4', 'MPL-5']))
class TestBPT(object):

    mechanisms = ['sf', 'comp', 'agn', 'seyfert', 'liner', 'invalid', 'ambiguous']

    @pytest.mark.parametrize('useoi', [(True), (False)], ids=['withoi', 'nooi'])
    def test_bpt(self, maps, useoi):
        if maps.bptsums is None:
            pytest.skip('no bpt data found in galaxy test data')

        bptflag = 'nooi' if useoi is False else 'global'

        masks, figure, axes = maps.get_bpt(show_plot=False, return_figure=True, use_oi=useoi)
        assert isinstance(figure, plt.Figure)

        for mech in self.mechanisms:
            assert mech in masks.keys()
            assert np.sum(masks[mech]['global']) == maps.bptsums[bptflag][mech]

        if useoi:
            assert len(axes) == 4
        else:
            assert len(axes) == 3

        for ax in axes:
            assert isinstance(ax, LocatableAxes)

    def test_bpt_diffsn(self, maps):

        if maps.bptsums is None:
            pytest.skip('no bpt data found in galaxy test data')

        masks, figure, __ = maps.get_bpt(show_plot=False, return_figure=True,
                                         use_oi=True, snr_min=5)
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
        assert record[0].message.args[0] == ('snr is deprecated. Use snr_min instead. '
                                             'snr will be removed in a future version of marvin')

        for mech in self.mechanisms:
            assert mech in masks.keys()
            assert np.sum(masks[mech]['global']) == maps.bptsums['snrmin5'][mech]

    def test_bind_to_figure(self, maps):

        __, __, axes = maps.get_bpt(show_plot=False, return_figure=True)

        assert len(axes) == 4

        for ax in axes:
            new_fig = ax.bind_to_figure()
            assert isinstance(new_fig, plt.Figure)
            assert new_fig.axes[0].get_ylabel() != ''

    def test_kewley_snr_warning(self, maps):

        with pytest.warns(MarvinDeprecationWarning) as record:
            bpt_kewley06(maps, snr=1, return_figure=False)

        assert len(record) == 1
        assert record[0].message.args[0] == ('snr is deprecated. Use snr_min instead. '
                                             'snr will be removed in a future version of marvin')

    def test_wrong_kw(self, maps):

        with pytest.raises(MarvinError) as error:
            maps.get_bpt(snr=5, return_figure=False, show_plot=False, extra_keyword='hola')

        with pytest.raises(MarvinError) as error2:
            bpt_kewley06(maps, snr=5, return_figure=False, extra_keyword='hola')

        assert 'unknown keyword extra_keyword' in str(error.value)
        assert 'unknown keyword extra_keyword' in str(error2.value)


class TestGetSNR(object):

    def test_get_snr_in_dict(self):

        assert get_snr({'ha': 5, 'hb': 4}, 'ha') == 5

    def test_get_snr_default(self):

        default = inspect.getargspec(get_snr).defaults[0]

        assert get_snr({'ha': 5, 'hb': 4}, 'xx') == default
