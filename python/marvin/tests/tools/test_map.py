#!/usr/bin/env python
# encoding: utf-8
#
# test_map.py
#
# @Author: Brett Andrews <andrews>
# @Date:   2017-07-02 13:08:00
# @Last modified by:   andrews
# @Last modified time: 2017-09-29 11:09:14

from copy import deepcopy

import numpy as np
import astropy
from astropy import units as u
import matplotlib
import pytest

from marvin import config
from marvin.core.exceptions import MarvinError
from marvin.utils.dap import datamodel
from marvin.tools.maps import Maps
from marvin.tools.map import Map, EnhancedMap
from marvin.tests import marvin_test_if
from marvin.utils.dap.datamodel.plotting import get_default_plot_params

value1 = np.array([[16.35, 0.8],
                   [0, -10.]])
value2 = np.array([[591., 1e-8],
                   [4., 10]])

value_prod12 = np.array([[9.66285000e+03, 8e-9],
                         [0, -100]])

ivar1 = np.array([[4, 1],
                  [6.97789734e+36, 1e8]])
ivar2 = np.array([[10, 1e-8],
                  [5.76744385e+36, 0]])

ivar_sum12 = np.array([[2.85714286e+00, 9.99999990e-09],
                       [3.15759543e+36, 0]])

ivar_prod12 = np.array([[1.10616234e-05, 1.56250000e-08],
                        [np.nan, 0.]])

ivar_pow_2 = np.array([[5.23472002e-08, 9.53674316e-01],
                      [np.inf, 25]])
ivar_pow_05 = np.array([[3.66072168e-03, 7.81250000e+00],
                       [np.inf, np.nan]])
ivar_pow_0 = np.array([[np.inf, np.inf],
                      [np.inf, np.inf]])
ivar_pow_m1 = np.array([[4, 1.],
                        [np.nan, 1e+08]])
ivar_pow_m2 = np.array([[2.67322500e+02, 1.6e-01],
                        [np.nan, 2.5e+09]])
ivar_pow_m05 = np.array([[0.97859327, 5],
                         [np.nan, np.nan]])

u_flux = u.erg / u.cm**2 / u.s / u.def_unit('spaxel')
u_flux2 = u_flux * u_flux


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


# @pytest.fixture(scope='function')
# def haflux(galaxy, data_origin):
#     maps = Maps(**_get_maps_kwargs(galaxy, data_origin))
#     ha = maps['emline_gflux_ha_6564']
#     ha.data_origin = data_origin
#     return ha


class TestMap(object):

    def test_map(self, map_, galaxy):

        assert map_.release == galaxy.release

        assert tuple(map_.shape) == tuple(galaxy.shape)
        assert map_.value.shape == tuple(galaxy.shape)
        assert map_.ivar.shape == tuple(galaxy.shape)
        assert map_.mask.shape == tuple(galaxy.shape)

        assert (map_.masked.data == map_.value).all()
        assert (map_.masked.mask == map_.mask.astype(bool)).all()

        assert pytest.approx(map_.snr, np.abs(map_.value * np.sqrt(map_.ivar)))

        assert datamodel[map_.maps._dapver][map_.property.full()].unit == map_.unit

        assert isinstance(map_.header, astropy.io.fits.header.Header)

    def test_plot(self, map_):
        fig, ax = map_.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes._subplots.Subplot)
        assert 'Make single panel map or one panel of multi-panel map plot.' in map_.plot.__doc__

    @marvin_test_if(mark='skip', map_={'data_origin': ['db']})
    def test_save_and_restore(self, temp_scratch, map_):

        fout = temp_scratch.join('test_map.mpf')
        map_.save(str(fout))
        assert fout.check() is True

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


class TestMapArith(object):

    @pytest.mark.parametrize('ivar1, ivar2, expected',
                             [(ivar1, ivar2, ivar_sum12)])
    def test_add_ivar(self, ivar1, ivar2, expected):
        assert pytest.approx(Map._add_ivar(ivar1, ivar2) == expected)

    @pytest.mark.parametrize('ivar1, ivar2, value1, value2, value_prod12, expected',
                             [(ivar1, ivar2, value1, value2, value_prod12, ivar_prod12)])
    def test_mul_ivar(self, ivar1, ivar2, value1, value2, value_prod12, expected):
        assert pytest.approx(Map._mul_ivar(ivar1, ivar2, value1, value2, value_prod12) == expected)

    @pytest.mark.parametrize('power, expected',
                             [(2, ivar_pow_2),
                              (0.5, ivar_pow_05),
                              (0, ivar_pow_0),
                              (-1, ivar_pow_m1),
                              (-2, ivar_pow_m2),
                              (-0.5, ivar_pow_m05)])
    @pytest.mark.parametrize('ivar, value,',
                             [(ivar1, value1)])
    def test_pow_ivar(self, ivar, value, power,expected):
        assert pytest.approx(Map._pow_ivar(ivar, value, power) == expected)

    @pytest.mark.parametrize('power', [2, 0.5, 0, -1, -2, -0.5])
    def test_pow_ivar_none(self, power):
        assert pytest.approx(Map._pow_ivar(None, np.arange(4), power) == np.zeros(4))

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

    def test_create_history(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        nii = maps['emline_gflux_nii_6585']
        ha = maps['emline_gflux_ha_6564']
        sii = maps['emline_gflux_sii_6732']
        n2ha = nii / ha
        s2ha = sii / ha

        expected_n2ha = '(emline_gflux_nii_6585 / emline_gflux_ha_6564)'
        assert Map._create_history(nii, ha, '/') == expected_n2ha

        expected_s2ha_nii = ('((emline_gflux_sii_6732 / emline_gflux_ha_6564) / '
                             'emline_gflux_nii_6585)')
        assert Map._create_history(s2ha, nii, '/') == expected_s2ha_nii

        expected_n2ha_s2ha = ('((emline_gflux_nii_6585 / emline_gflux_ha_6564) / '
                              '(emline_gflux_sii_6732 / emline_gflux_ha_6564))')
        assert Map._create_history(n2ha, s2ha, '/') == expected_n2ha_s2ha

    def test_create_parents(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        nii = maps['emline_gflux_nii_6585']
        ha = maps['emline_gflux_ha_6564']
        n2ha = nii / ha

        assert Map._create_parents(nii, ha) == [nii, ha]
        assert Map._create_parents(n2ha, ha) == [[nii, ha], ha]
        assert Map._create_parents(n2ha, n2ha) == [[nii, ha], [nii, ha]]

    @pytest.mark.parametrize('property1, channel1, property2, channel2',
                             [('emline_gflux', 'ha_6564', 'emline_gflux', 'nii_6585'),
                              ('emline_gvel', 'ha_6564', 'stellar_vel', None)])
    def test_add_maps(self, galaxy, property1, channel1, property2, channel2):
        maps = Maps(plateifu=galaxy.plateifu)
        map1 = maps.getMap(property_name=property1, channel=channel1)
        map2 = maps.getMap(property_name=property2, channel=channel2)
        map12 = map1 + map2

        assert pytest.approx(map12.value == map1.value + map2.value)
        assert pytest.approx(map12.ivar == map1._add_ivar(map1.ivar, map2.ivar))
        assert pytest.approx(map12.mask == map1.mask & map2.mask)

    @pytest.mark.parametrize('property1, channel1, property2, channel2',
                             [('emline_gflux', 'ha_6564', 'emline_gflux', 'nii_6585'),
                              ('emline_gvel', 'ha_6564', 'stellar_vel', None)])
    def test_subtract_maps(self, galaxy, property1, channel1, property2, channel2):
        maps = Maps(plateifu=galaxy.plateifu)
        map1 = maps.getMap(property_name=property1, channel=channel1)
        map2 = maps.getMap(property_name=property2, channel=channel2)
        map12 = map1 - map2

        assert pytest.approx(map12.value == map1.value - map2.value)
        assert pytest.approx(map12.ivar == map1._add_ivar(map1.ivar, map2.ivar))
        assert pytest.approx(map12.mask == map1.mask & map2.mask)

    @pytest.mark.parametrize('property1, channel1, property2, channel2',
                             [('emline_gflux', 'ha_6564', 'emline_gflux', 'nii_6585'),
                              ('emline_gvel', 'ha_6564', 'stellar_vel', None)])
    def test_multiply_maps(self, galaxy, property1, channel1, property2, channel2):
        maps = Maps(plateifu=galaxy.plateifu)
        map1 = maps.getMap(property_name=property1, channel=channel1)
        map2 = maps.getMap(property_name=property2, channel=channel2)
        map12 = map1 * map2

        assert pytest.approx(map12.value == map1.value * map2.value)
        assert pytest.approx(map12.ivar == map1._mul_ivar(map1.ivar, map2.ivar, map1.value,
                                                          map2.value, map12.value))
        assert pytest.approx(map12.mask == map1.mask & map2.mask)

    @pytest.mark.parametrize('property1, channel1, property2, channel2',
                             [('emline_gflux', 'ha_6564', 'emline_gflux', 'nii_6585'),
                              ('emline_gvel', 'ha_6564', 'stellar_vel', None)])
    def test_divide_maps(self, galaxy, property1, channel1, property2, channel2):
        maps = Maps(plateifu=galaxy.plateifu)
        map1 = maps.getMap(property_name=property1, channel=channel1)
        map2 = maps.getMap(property_name=property2, channel=channel2)
        map12 = map1 / map2

        with np.errstate(divide='ignore', invalid='ignore'):
            assert pytest.approx(map12.value == map1.value / map2.value)

        assert pytest.approx(map12.ivar == map1._mul_ivar(map1.ivar, map2.ivar, map1.value,
                                                          map2.value, map12.value))
        assert pytest.approx(map12.mask == map1.mask & map2.mask)

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

        assert pytest.approx(map_new.value == map_orig.value**power)
        assert pytest.approx(map_new.ivar == ivar_new)
        assert (map_new.mask == map_orig.mask).all()

    @marvin_test_if(mark='skip', galaxy=dict(release=['MPL-4']))
    def test_stellar_sigma_correction(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        stsig = maps['stellar_sigma']
        stsigcorr = maps['stellar_sigmacorr']
        expected = (stsig**2 - stsigcorr**2)**0.5
        actual = stsig.inst_sigma_correction()
        assert pytest.approx(actual.value == expected.value)
        assert pytest.approx(actual.ivar == expected.ivar)
        assert (actual.mask == expected.mask).all()

    @marvin_test_if(mark='include', galaxy=dict(release=['MPL-4']))
    def test_stellar_sigma_correction_MPL4(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        stsig = maps['stellar_sigma']
        with pytest.raises(MarvinError) as ee:
            stsig.inst_sigma_correction()

        assert 'Instrumental broadening correction not implemented for MPL-4.' in str(ee.value)

    def test_stellar_sigma_correction_invalid_property(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        ha = maps['emline_gflux_ha_6564']

        with pytest.raises(MarvinError) as ee:
            ha.inst_sigma_correction()

        assert ('Cannot correct {0}_{1} '.format(ha.property.name, ha.channel) +
                'for instrumental broadening.') in str(ee.value)

    def test_emline_sigma_correction(self, galaxy):
        maps = Maps(plateifu=galaxy.plateifu)
        hasig = maps['emline_gsigma_ha_6564']
        emsigcorr = maps['emline_instsigma_ha_6564']

        expected = (hasig**2 - emsigcorr**2)**0.5
        actual = hasig.inst_sigma_correction()

        assert pytest.approx(actual.value == expected.value)
        assert pytest.approx(actual.ivar == expected.ivar)
        assert (actual.mask == expected.mask).all()


class TestMaskbit(object):
    
    def test_masked(self, map_release_only):
        __, dapver = config.lookUpVersions(map_release_only.release)
        params = get_default_plot_params(dapver)
        expected = map_release_only.get_mask(bitnames=params['default']['bitmasks'], dtype=bool)

        assert pytest.approx(map_release_only.masked.data, map_release_only.value)
        assert (map_release_only.masked.mask == expected).all()

    def test_get_mask_no_bitnames(self, map_release_only):

        actual = map_release_only.get_mask()

        __, dapver = config.lookUpVersions(map_release_only.release)
        params = get_default_plot_params(dapver)
        expected = map_release_only.get_mask(bitnames=params['default']['bitmasks'])

        assert (actual == expected).all()

    def test_get_mask_one_bitname_as_string(self, map_release_only):
        actual = map_release_only.get_mask(bitnames='DONOTUSE')
        expected = map_release_only.mask & 2**map_release_only.maskbit.bits['DONOTUSE'].value
        assert (actual == expected).all()

    def test_get_mask_one_bitname_as_list(self, map_release_only):
        actual = map_release_only.get_mask(bitnames=('DONOTUSE',))
        expected = map_release_only.mask & 2**map_release_only.maskbit.bits['DONOTUSE'].value
        assert (actual == expected).all()
    
    @marvin_test_if(mark='skip', map_release_only=dict(release=['MPL-4']))
    def test_get_mask_multiple_bitnames(self, map_release_only):
        actual = map_release_only.get_mask(bitnames=('NOCOV', 'DONOTUSE'))

        mask0 = map_release_only.mask & 2**map_release_only.maskbit.bits['NOCOV'].value
        mask1 = map_release_only.mask & 2**map_release_only.maskbit.bits['DONOTUSE'].value
        expected = np.sum((mask0, mask1), axis=0, dtype=int)

        assert (actual == expected).all()

    @marvin_test_if(mark='include', map_release_only=dict(release=['MPL-4']))
    @pytest.mark.parametrize('output, expected',
                             [('name', ['DONOTUSE']),
                              ('value', [0])])
    def test_get_bits_mpl4(self, map_release_only, output, expected):
        assert map_release_only.get_bits(1, output=output) == expected

    @marvin_test_if(mark='skip', map_release_only=dict(release=['MPL-4']))
    @pytest.mark.parametrize('output, expected',
                             [('name', ['NOCOV', 'LOWCOV']),
                              ('value', [0, 1])])
    def test_get_bits(self, map_release_only, output, expected):
        assert map_release_only.get_bits(3, output=output) == expected

    @marvin_test_if(mark='include', map_release_only=dict(release=['MPL-4']))
    def test_get_bits_object_mpl4(self, map_release_only):
        actual = map_release_only.get_bits(1, output='object')
        expected = [map_release_only.maskbit.bits['DONOTUSE']]
        assert actual == expected

    @marvin_test_if(mark='skip', map_release_only=dict(release=['MPL-4']))
    def test_get_bits_object(self, map_release_only):
        actual = map_release_only.get_bits(3, output='object')
        expected = [map_release_only.maskbit.bits['NOCOV'],
                    map_release_only.maskbit.bits['LOWCOV']]
        assert actual == expected

    def test_get_bits_zero(self, map_release_only):
        assert map_release_only.get_bits(0, output='object') == []
    
    @marvin_test_if(mark='include', map_release_only=dict(release=['MPL-4']))
    @pytest.mark.parametrize('names, expected',
                             [('DONOTUSE', 1)])
    def test_get_bit_int_mpl4(self, map_release_only, names, expected):
        assert map_release_only.get_bit_int(names) == expected

    @marvin_test_if(mark='skip', map_release_only=dict(release=['MPL-4']))
    @pytest.mark.parametrize('names, expected',
                             [(['NOCOV', 'LOWCOV'], 3),
                              ('DONOTUSE', 1073741824)])
    def test_get_bit_int_mpl4(self, map_release_only, names, expected):
        assert map_release_only.get_bit_int(names) == expected



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
