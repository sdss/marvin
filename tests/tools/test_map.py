#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Brian Cherinka, José Sánchez-Gallego, and Brett Andrews
# @Date: 2017-07-02
# @Filename: test_map.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by:   andrews
# @Last modified time: 2019-11-22 12:11:29


import operator
from copy import deepcopy

import matplotlib
import numpy as np
import pytest
from astropy import units as u

from marvin.core.exceptions import MarvinError
from tests import marvin_test_if
from marvin.tools.maps import Maps
from marvin.tools.quantities import EnhancedMap, Map
from marvin.utils.datamodel.dap import datamodel
from marvin.utils.general.maskbit import Maskbit


value1 = np.array([[16.35, 0.8],
                   [0, -10.]])
value2 = np.array([[591., 1e-8],
                   [4., 10]])

value_prod12 = np.array([[9.66285000e+03, 8e-9],
                         [0, -100]])

value_log2 = np.array([[2.77158748, -8.],
                       [0.60205999, 1.]])

ivar1 = np.array([[4, 1],
                  [6.97789734e+36, 1e8]])
ivar2 = np.array([[10, 1e-8],
                  [5.76744385e+36, 0]])

ivar_sum12 = np.array([[2.85714286e+00, 9.99999990e-09],
                       [3.15759543e+36, 0]])

ivar_prod12 = np.array([[1.10616234e-05, 1.56250000e-08],
                        [0, 0.]])

ivar_pow_2 = np.array([[5.23472002e-08, 9.53674316e-01],
                      [0, 25]])
ivar_pow_05 = np.array([[3.66072168e-03, 7.81250000e+00],
                       [0, 0]])
ivar_pow_0 = np.array([[0, 0],
                      [0, 0]])
ivar_pow_m1 = np.array([[4, 1.],
                        [0, 1e+08]])
ivar_pow_m2 = np.array([[2.67322500e+02, 1.6e-01],
                        [0, 2.5e+09]])
ivar_pow_m05 = np.array([[0.97859327, 5],
                         [0, 0]])

ivar_log1 = np.array([[3.67423420e-04, 4.34294482e+07],
                      [4.11019127e-20, 4.34294482e-06]])

u_flux = u.erg / u.cm**2 / u.s / u.def_unit('spaxel')
u_flux2 = u_flux * u_flux

ufuncs = [it for it in dir(np) if isinstance(getattr(np, it), np.ufunc)]


def _get_maps_kwargs(galaxy, data_origin):

    if data_origin == 'file':
        maps_kwargs = dict(filename=galaxy.mapspath)
    else:
        maps_kwargs = dict(plateifu=galaxy.plateifu, release=galaxy.release,
                           bintype=galaxy.bintype, template_kin=galaxy.template,
                           mode='local' if data_origin == 'db' else 'remote')

    return maps_kwargs


@pytest.fixture(scope='function', params=[('emline_gflux', 'ha_6564'),
                                          ('emline_gvel', 'oiii_5008'),
                                          ('stellar_vel', None),
                                          ('stellar_sigma', None)])
def map_(request, galaxy, data_origin):
    maps = Maps(**_get_maps_kwargs(galaxy, data_origin))
    map_ = maps.getMap(property_name=request.param[0], channel=request.param[1])
    map_.data_origin = data_origin
    return map_


class TestMap(object):

    def test_map(self, map_, galaxy):

        assert map_.getMaps().release == galaxy.release

        assert tuple(map_.shape) == tuple(galaxy.shape)
        assert map_.value.shape == tuple(galaxy.shape)
        assert map_.ivar.shape == tuple(galaxy.shape)
        assert map_.mask.shape == tuple(galaxy.shape)

        assert (map_.masked.data == map_.value).all()
        assert (map_.masked.mask == map_.mask.astype(bool)).all()

        assert map_.snr == pytest.approx(np.abs(map_.value * np.sqrt(map_.ivar)))

        assert datamodel[map_.getMaps()._dapver][map_.datamodel.full()].unit == map_.unit

    def test_plot(self, map_):
        fig, ax = map_.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes._subplots.Subplot)
        assert 'Make single panel map or one panel of multi-panel map plot.' in map_.plot.__doc__

    @marvin_test_if(mark='skip', map_={'data_origin': ['db']})
    def test_save_and_restore(self, temp_scratch, map_):

        fout = temp_scratch / 'test_map.mpf'
        map_.save(str(fout))
        assert fout.exists() is True

        map_restored = Map.restore(str(fout), delete=True)
        assert tuple(map_.shape) == tuple(map_restored.shape)

    @pytest.mark.parametrize('property_name, channel',
                             [('emline_gflux', 'ha_6564'),
                              ('stellar_vel', None)])
    def test_deepcopy(self, galaxy, property_name, channel):
        maps = Maps(plateifu=galaxy.plateifu)
        map1 = maps.getMap(property_name=property_name, channel=channel)
        map2 = deepcopy(map1)

        for attr in vars(map1):
            if not attr.startswith('_'):
                value = getattr(map1, attr)
                value2 = getattr(map2, attr)

                if isinstance(value, np.ndarray):
                    assert np.isclose(value, value2).all()

                elif isinstance(value, np.ma.core.MaskedArray):
                    assert (np.isclose(value.data, value2.data).all() and
                            (value.mask == value2.mask).all())

                elif isinstance(value, Maskbit) or isinstance(value[0], Maskbit):

                    if isinstance(value, Maskbit):
                        value = [value]
                        value2 = [value2]

                    for mb, mb2 in zip(value, value2):
                        for it in ['bits', 'description', 'labels', 'mask', 'name']:
                            assert getattr(mb, it) == getattr(mb2, it)

                        assert (mb.schema == mb2.schema).all().all()

                elif isinstance(value, Maps):
                    pass

                else:
                    assert value == value2, attr

    def test_getMap_invalid_property(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        with pytest.raises(ValueError) as ee:
            maps.getMap(property_name='mythical_property')

        assert 'Your input value is too ambiguous.' in str(ee.value)

    def test_getMap_invalid_channel(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        with pytest.raises(ValueError) as ee:
            maps.getMap(property_name='emline_gflux', channel='mythical_channel')

        assert 'Your input value is too ambiguous.' in str(ee.value)

    @marvin_test_if(mark='include', maps={'plateifu': '8485-1901',
                                          'release': 'DR17',
                                          'mode': 'local',
                                          'data_origin': 'file'})
    def test_quatities_reorder(self, maps):
        """Asserts the unit survives a quantity reorder (issue #374)."""

        ha = maps['emline_gflux_ha']

        assert ha is not None
        assert ha.unit is not None

        reordered_ha = np.moveaxis(ha, 0, -1)
        assert reordered_ha.unit is not None

    @marvin_test_if(mark='include', maps={'plateifu': '8485-1901',
                                          'release': 'DR17',
                                          'bintype': ['HYB10']})
    def test_get_spaxel(self, maps):
        """Tests `.Map.getSpaxel`."""

        ha = maps['emline_gflux_ha']

        spaxel = ha.getSpaxel(x=10, y=10, xyorig='lower')

        assert spaxel is not None
        assert spaxel.x == 10 and spaxel.y == 10

    @marvin_test_if(mark='skip', galaxy=dict(release=['DR17']))
    def test_stellar_sigma_values(self, maps, galaxy):
        ''' Assert values for stellar_sigma and stellar_sigmacorr are different (issue #411) '''

        ss = maps.stellar_sigma
        sc = maps.stellar_sigmacorr
        compare = sum(ss.value == sc.value)
        assert len(np.unique(compare)) > 1
        x = galaxy.dap['x']
        y = galaxy.dap['y']
        ssvalue = galaxy.dap['stellar_sigma'][galaxy.bintype.name]
        scvalue = galaxy.dap['stellar_sigmacorr'][galaxy.bintype.name]
        assert ssvalue == pytest.approx(ss[x, y].value, 1e-4)
        assert scvalue == pytest.approx(sc[x, y].value, 1e-4)

    def test_datamodel(self, maps):

        gew_ha = maps.emline_gew_ha_6564
        assert gew_ha.datamodel.description == ('Gaussian-fitted equivalent widths measurements '
                                                '(based on EMLINE_GFLUX). Channel = H-alpha 6564.')
    @marvin_test_if(mark='include', galaxy=dict(release=['DR15']))
    def test_stellar_sigma_mpl6(self, maps, galaxy):
        with pytest.raises(MarvinError) as cm:
            __ = maps.stellar_sigmacorr
        assert 'stellar_sigmacorr is unreliable in DR15. Please use DR17.' in str(cm.value)


class TestMapArith(object):

    def test_add_constant(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']
        ha10 = ha + 10.

        assert ha10.value == pytest.approx(ha.value + 10.)
        assert ha10.ivar == pytest.approx(ha.ivar)
        assert ha10.mask == pytest.approx(ha.mask)

    def test_reflexive_add_constant(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']
        ha10 = 10. + ha

        assert ha10.value == pytest.approx(ha.value + 10.)
        assert ha10.ivar == pytest.approx(ha.ivar)
        assert ha10.mask == pytest.approx(ha.mask)

    def test_subtract_constant(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']
        ha10 = ha - 10.

        assert ha10.value == pytest.approx(ha.value - 10.)
        assert ha10.ivar == pytest.approx(ha.ivar)
        assert ha10.mask == pytest.approx(ha.mask)

    def test_reflexive_subtract_constant(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']
        ha10 = 10. - ha

        assert ha10.value == pytest.approx(10. - ha.value)
        assert ha10.ivar == pytest.approx(ha.ivar)
        assert ha10.mask == pytest.approx(ha.mask)

    def test_multiply_constant(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']
        ha10 = ha * 10.

        assert ha10.value == pytest.approx(ha.value * 10.)
        assert ha10.ivar == pytest.approx(ha.ivar / 10.**2)
        assert ha10.mask == pytest.approx(ha.mask)

    def test_reflexive_multiply_constant(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']
        ha10 = 10. * ha

        assert ha10.value == pytest.approx(ha.value * 10.)
        assert ha10.ivar == pytest.approx(ha.ivar / 10.**2)
        assert ha10.mask == pytest.approx(ha.mask)

    def test_divide_constant(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']
        ha10 = ha / 10.

        assert ha10.value == pytest.approx(ha.value / 10.)
        assert ha10.ivar == pytest.approx(ha.ivar * 10.**2)
        assert ha10.mask == pytest.approx(ha.mask)

    def test_reflexive_divide_constant(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']
        ha10 = 10. / ha

        assert ha10.value == pytest.approx(10. / ha.value)
        assert ha10.ivar == pytest.approx(ha.ivar)
        assert ha10.mask == pytest.approx(ha.mask)

    @pytest.mark.parametrize('ivar1, ivar2, expected',
                             [(ivar1, ivar2, ivar_sum12)])
    def test_add_ivar(self, ivar1, ivar2, expected):
        assert Map._add_ivar(ivar1, ivar2) == pytest.approx(expected)

    @pytest.mark.parametrize('ivar1, ivar2, value1, value2, value_prod12, expected',
                             [(ivar1, ivar2, value1, value2, value_prod12, ivar_prod12)])
    def test_mul_ivar(self, ivar1, ivar2, value1, value2, value_prod12, expected):
        ivar = Map._mul_ivar(ivar1, ivar2, value1, value2, value_prod12)
        ivar[np.isnan(ivar)] = 0
        ivar[np.isinf(ivar)] = 0
        assert ivar == pytest.approx(expected)

    @pytest.mark.parametrize('power, expected',
                             [(2, ivar_pow_2),
                              (0.5, ivar_pow_05),
                              (0, ivar_pow_0),
                              (-1, ivar_pow_m1),
                              (-2, ivar_pow_m2),
                              (-0.5, ivar_pow_m05)])
    @pytest.mark.parametrize('ivar, value,',
                             [(ivar1, value1)])
    def test_pow_ivar(self, ivar, value, power, expected):
        ivar = Map._pow_ivar(ivar, value, power)
        ivar[np.isnan(ivar)] = 0
        ivar[np.isinf(ivar)] = 0
        assert ivar == pytest.approx(expected)

    @pytest.mark.parametrize('power', [2, 0.5, 0, -1, -2, -0.5])
    def test_pow_ivar_none(self, power):
        assert Map._pow_ivar(None, np.arange(4), power) == pytest.approx(np.zeros(4))

    @pytest.mark.parametrize('ivar, value, expected',
                             [(ivar1, value2, ivar_log1)])
    def test_log10_ivar(self, ivar, value, expected):
        actual = Map._log10_ivar(ivar, value)
        assert actual == pytest.approx(expected)

    def test_log10(self, maps_release_only):
        niiha = maps_release_only.emline_gflux_nii_6585 / maps_release_only.emline_gflux_nii_6585
        log_niiha = np.log10(niiha)
        ivar = np.log10(np.e) * niiha.ivar**-0.5 / niiha.value

        assert log_niiha.value == pytest.approx(np.log10(niiha.value), nan_ok=True)
        assert log_niiha.ivar == pytest.approx(ivar, nan_ok=True)
        assert (log_niiha.mask == niiha.mask).all()
        assert log_niiha.unit == u.dimensionless_unscaled

    @pytest.mark.runslow
    @marvin_test_if(mark='skip', ufunc=['log10'])
    @pytest.mark.parametrize('ufunc', ufuncs)
    def test_np_ufunc_notimplemented(self, maps_release_only, ufunc):
        ha = maps_release_only.emline_gflux_ha_6564
        nii = maps_release_only.emline_gflux_nii_6585

        with pytest.raises(NotImplementedError) as ee:
            if getattr(getattr(np, ufunc), 'nargs') <= 2:
                getattr(np, ufunc)(ha)

            else:
                getattr(np, ufunc)(nii, ha)

        expected = 'np.{0} is not implemented for Map.'.format(getattr(np, ufunc).__name__)
        assert str(ee.value) == expected

    @pytest.mark.parametrize('unit1, unit2, op, expected',
                             [(u_flux, u_flux, '+', u_flux),
                              (u_flux, u_flux, '-', u_flux),
                              (u_flux, u_flux, '*', u_flux2),
                              (u_flux, u_flux, '/', u.dimensionless_unscaled),
                              (u.km, u.s, '*', u.km * u.s),
                              (u.km, u.s, '/', u.km / u.s)])
    def test_unit_propagation(self, unit1, unit2, op, expected):
        assert Map._unit_propagation(unit1, unit2, op) == expected

    @pytest.mark.parametrize('unit1, unit2, op',
                             [(u_flux, u.km, '+'),
                              (u_flux, u.km, '-')])
    def test_unit_propagation_mismatch(self, unit1, unit2, op):
        with pytest.warns(UserWarning):
            assert Map._unit_propagation(unit1, unit2, op) is None

    @pytest.mark.parametrize('property1, channel1, property2, channel2',
                             [('emline_gflux', 'ha_6564', 'emline_gflux', 'nii_6585'),
                              ('emline_gvel', 'ha_6564', 'stellar_vel', None)])
    def test_add_maps(self, galaxy, property1, channel1, property2, channel2):
        maps = Maps(plateifu=galaxy.plateifu)
        map1 = maps.getMap(property_name=property1, channel=channel1)
        map2 = maps.getMap(property_name=property2, channel=channel2)
        map12 = map1 + map2

        assert map12.value == pytest.approx(map1.value + map2.value)
        assert map12.ivar == pytest.approx(map1._add_ivar(map1.ivar, map2.ivar))
        assert map12.mask == pytest.approx(map1.mask | map2.mask)

    @pytest.mark.parametrize('property1, channel1, property2, channel2',
                             [('emline_gflux', 'ha_6564', 'emline_gflux', 'nii_6585'),
                              ('emline_gvel', 'ha_6564', 'stellar_vel', None)])
    def test_subtract_maps(self, galaxy, property1, channel1, property2, channel2):
        maps = Maps(plateifu=galaxy.plateifu)
        map1 = maps.getMap(property_name=property1, channel=channel1)
        map2 = maps.getMap(property_name=property2, channel=channel2)
        map12 = map1 - map2

        assert map12.value == pytest.approx(map1.value - map2.value)
        assert map12.ivar == pytest.approx(map1._add_ivar(map1.ivar, map2.ivar))
        assert map12.mask == pytest.approx(map1.mask | map2.mask)

    @pytest.mark.parametrize('property1, channel1, property2, channel2',
                             [('emline_gflux', 'ha_6564', 'emline_gflux', 'nii_6585'),
                              ('emline_gvel', 'ha_6564', 'stellar_vel', None)])
    def test_multiply_maps(self, galaxy, property1, channel1, property2, channel2):
        maps = Maps(plateifu=galaxy.plateifu)
        map1 = maps.getMap(property_name=property1, channel=channel1)
        map2 = maps.getMap(property_name=property2, channel=channel2)
        map12 = map1 * map2

        ivar = map1._mul_ivar(map1.ivar, map2.ivar, map1.value, map2.value, map12.value)
        ivar[np.isnan(ivar)] = 0
        ivar[np.isinf(ivar)] = 0

        assert map12.value == pytest.approx(map1.value * map2.value)
        assert map12.ivar == pytest.approx(ivar)
        assert map12.mask == pytest.approx(map1.mask | map2.mask)

    @pytest.mark.parametrize('property1, channel1, property2, channel2',
                             [('emline_gflux', 'ha_6564', 'emline_gflux', 'nii_6585'),
                              ('emline_gvel', 'ha_6564', 'stellar_vel', None)])
    def test_divide_maps(self, galaxy, property1, channel1, property2, channel2):
        maps = Maps(plateifu=galaxy.plateifu)
        map1 = maps.getMap(property_name=property1, channel=channel1)
        map2 = maps.getMap(property_name=property2, channel=channel2)
        map12 = map1 / map2

        ivar = map1._mul_ivar(map1.ivar, map2.ivar, map1.value, map2.value, map12.value)
        ivar[np.isnan(ivar)] = 0
        ivar[np.isinf(ivar)] = 0

        mask = map1.mask | map2.mask
        bad = np.isnan(map12.value) | np.isinf(map12.value)
        mask[bad] = mask[bad] | map12.pixmask.labels_to_value('DONOTUSE')

        with np.errstate(divide='ignore', invalid='ignore'):
            assert map12.value == pytest.approx(map1.value / map2.value, nan_ok=True)

        assert map12.ivar == pytest.approx(ivar)
        assert map12.mask == pytest.approx(mask)

    @pytest.mark.runslow
    @pytest.mark.parametrize('power', [2, 0.5, 0, -1, -2, -0.5])
    @pytest.mark.parametrize('property_name, channel',
                             [('emline_gflux', 'ha_6564'),
                              ('stellar_vel', None)])
    def test_pow(self, galaxy, property_name, channel, power):
        maps = Maps(plateifu=galaxy.plateifu)
        map_orig = maps.getMap(property_name=property_name, channel=channel)
        map_new = map_orig**power

        sig_orig = np.sqrt(1. / map_orig.ivar)
        sig_new = map_new.value * power * sig_orig * map_orig.value
        ivar_new = 1 / sig_new**2.
        ivar_new[np.isnan(ivar_new)] = 0
        ivar_new[np.isinf(ivar_new)] = 0

        assert map_new.value == pytest.approx(map_orig.value**power, nan_ok=True)
        assert map_new.ivar == pytest.approx(ivar_new)
        assert (map_new.mask == map_orig.mask).all()

    @marvin_test_if(mark='skip', galaxy=dict(release=['MPL-4', 'MPL-6']))
    def test_stellar_sigma_correction(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        stsig = maps['stellar_sigma']
        stsigcorr = maps['stellar_sigmacorr']

        expected = (stsig**2 - stsigcorr**2)**0.5
        expected.ivar = (expected.value / stsig.value) * stsig.ivar
        expected.ivar[stsig.ivar == 0] = 0
        expected.ivar[stsigcorr.value >= stsig.value] = 0
        expected.value[stsigcorr.value >= stsig.value] = 0

        actual = stsig.inst_sigma_correction()

        assert actual.value == pytest.approx(expected.value, nan_ok=True)
        assert actual.ivar == pytest.approx(expected.ivar)
        assert (actual.mask == expected.mask).all()
        assert actual.datamodel == stsig.datamodel

    @marvin_test_if(mark='include', galaxy=dict(release=['MPL-4', 'MPL-6']))
    def test_stellar_sigma_correction_MPL4(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        stsig = maps['stellar_sigma']

        if galaxy.release == 'MPL-4':
            errmsg = 'Instrumental broadening correction not implemented for MPL-4.'
        elif galaxy.release == 'MPL-6':
            errmsg = 'The stellar sigma corrections in MPL-6 are unreliable. Please use MPL-7.'

        with pytest.raises(MarvinError) as ee:
            stsig.inst_sigma_correction()

        assert errmsg in str(ee.value)

    def test_stellar_sigma_correction_invalid_property(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']

        with pytest.raises(MarvinError) as ee:
            ha.inst_sigma_correction()

        assert ('Cannot correct {0}_{1} '.format(ha.datamodel.name, ha.datamodel.channel) +
                'for instrumental broadening.') in str(ee.value)

    def test_emline_sigma_correction(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        hasig = maps['emline_gsigma_ha_6564']
        emsigcorr = maps['emline_instsigma_ha_6564']

        expected = (hasig**2 - emsigcorr**2)**0.5
        expected.ivar = (expected.value / hasig.value) * hasig.ivar
        expected.ivar[hasig.ivar == 0] = 0
        expected.ivar[emsigcorr.value >= hasig.value] = 0
        expected.value[emsigcorr.value >= hasig.value] = 0

        actual = hasig.inst_sigma_correction()

        assert actual.value == pytest.approx(expected.value, nan_ok=True)
        assert actual.ivar == pytest.approx(expected.ivar)
        assert (actual.mask == expected.mask).all()
        assert actual.datamodel == hasig.datamodel

    @marvin_test_if(mark='skip', galaxy=dict(release=['MPL-4', 'MPL-5']))
    @pytest.mark.parametrize('channel, op',
                             [('hb', '*'),
                              ('d4000', '*'),
                              ('cn1', '+'),
                              ])
    def test_specindex_sigma_correction(self, galaxy, channel, op):
        maps = Maps(plateifu=galaxy.plateifu)
        si = maps['specindex_' + channel]
        sicorr = maps['specindex_corr' + channel]

        ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
        expected = ops[op](si, sicorr)

        actual = si.specindex_correction()

        assert actual.value == pytest.approx(expected.value, nan_ok=True)
        assert actual.ivar == pytest.approx(expected.ivar)
        assert (actual.mask == expected.mask).all()
        assert actual.datamodel == si.datamodel


class TestMaskbit(object):

    def test_masked(self, maps_release_only):
        params = maps_release_only.datamodel.parent.get_default_plot_params()
        ha = maps_release_only['emline_gflux_ha_6564']
        expected = ha.pixmask.get_mask(params['default']['bitmasks'], dtype=bool)

        assert ha.masked.data == pytest.approx(ha.value)
        assert (ha.masked.mask == expected).all()

    @marvin_test_if(mark='include', maps_release_only=dict(release=['MPL-4']))
    def test_values_to_bits_mpl4(self, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert ha.pixmask.values_to_bits(1) == [0]

    @marvin_test_if(mark='skip', maps_release_only=dict(release=['MPL-4']))
    def test_values_to_bits(self, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert ha.pixmask.values_to_bits(3) == [0, 1]

    @marvin_test_if(mark='include', maps_release_only=dict(release=['MPL-4']))
    def test_values_to_labels_mpl4(self, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert ha.pixmask.values_to_labels(1) == ['DONOTUSE']

    @marvin_test_if(mark='skip', maps_release_only=dict(release=['MPL-4']))
    def test_values_to_labels(self, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert ha.pixmask.values_to_labels(3) == ['NOCOV', 'LOWCOV']

    @marvin_test_if(mark='include', maps_release_only=dict(release=['MPL-4']))
    def test_labels_to_value_mpl4(self, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert ha.pixmask.labels_to_value('DONOTUSE') == 1

    @marvin_test_if(mark='skip', maps_release_only=dict(release=['MPL-4']))
    @pytest.mark.parametrize('names, expected',
                             [(['NOCOV', 'LOWCOV'], 3),
                              ('DONOTUSE', 1073741824)])
    def test_labels_to_value(self, maps_release_only, names, expected):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert ha.pixmask.labels_to_value(names) == expected

    @pytest.mark.parametrize('flag',
                             ['manga_target1',
                              'manga_target2',
                              'manga_target3',
                              'target_flags',
                              'pixmask'])
    def test_flag(self, flag, maps_release_only):
        ha = maps_release_only['emline_gflux_ha_6564']
        assert getattr(ha, flag, None) is not None


class TestEnhancedMap(object):

    def test_overridden_methods(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']
        nii = maps['emline_gflux_nii_6585']
        n2ha = nii / ha

        assert isinstance(n2ha, EnhancedMap)

        methods = ['_init_map_from_maps', '_get_from_file', '_get_from_db', '_get_from_api',
                   'inst_sigma_correction']

        for method in methods:
            with pytest.raises(AttributeError) as ee:
                meth = getattr(n2ha, method)
                meth()

            assert "'EnhancedMap' has no attribute '{}'.".format(method) in str(ee.value)
