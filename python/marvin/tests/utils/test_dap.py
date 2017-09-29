#!/usr/bin/env python3
# encoding: utf-8
#
# test_dap_pytest.py
#
# Created by José Sánchez-Gallego on 19 Sep 2016.
# Adpated to pytest by Brett Andrews on 15 Jun 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from collections import OrderedDict

import pytest
import astropy

from marvin.utils.dap.datamodel import Bit, Maskbit
# from marvin.utils.dap.datamodel import MapsProperty, MapsPropertyList, get_dap_datamodel
# from marvin.tests import UseReleases
# from marvin import config

# @UseReleases('MPL-5')
# class TestMapsProperties(object):
#
#     def test_dap_datamodel_mpl4(self, release):
#         datamodel = get_dap_datamodel('1.1.1')
#         assert len(datamodel) == 10
#         assert datamodel.version == '1.1.1'
#         assert isinstance(datamodel, MapsPropertyList)
#         assert isinstance(datamodel[0], MapsProperty)
#
#     def test_MapsPropertyList(self, release):
#         drpver, dapver = config.lookUpVersions(release)
#         datamodel = get_dap_datamodel(dapver)
#         assert datamodel.version == dapver
#         assert 'EMLINE_GFLUX' in datamodel
#         assert not ('emline_bad' in datamodel)
#         assert isinstance(datamodel['emline_gflux'], MapsProperty)
#         assert isinstance(datamodel == 'emline_gflux', MapsProperty)
#         assert (datamodel == 'emline_bad') is None, MapsProperty
#
#     def test_MapsPropertyList_get(self, release):
#         drpver, dapver = config.lookUpVersions(release)
#         datamodel = get_dap_datamodel(dapver)
#         assert datamodel.get('badname_badchannel') is None
#         assert datamodel.get('emline_gflux') is None
#         assert datamodel.get('emline_gflux_badchannel') is None
#
#         maps_prop, channel = datamodel.get('emline_gflux_ha_6564')
#         assert isinstance(maps_prop, MapsProperty)
#         assert maps_prop.name == 'emline_gflux'
#         assert channel == 'ha_6564'

bits = OrderedDict([
    ('BITZERO', Bit(0, 'BITZERO', 'The zeroth bit.')),
    ('BITONE', Bit(1, 'BITONE', 'The first bit.'))
])
name = 'MYMASK'
description = 'My first Maskbit.'

class TestBit(object):

    def test_bit_init(self):
        value = 0
        name = 'firstbit'
        description = 'The first bit.'
        firstbit = Bit(value=value, name=name, description=description)

        assert firstbit.value == value
        assert firstbit.name == name
        assert firstbit.description == description
        assert str(firstbit) == "<Bit  0 name='firstbit'>"


class TestMaskbit(object):

    def test_maskbit_init(self, bits=bits, name=name, description=description):

        mb = Maskbit(bits=bits, name=name, description=description)

        assert mb.bits == bits
        assert mb.name == name
        assert mb.description == description
        assert str(mb) == "<Maskbit name='MYMASK'>"

    def test_maskbit_to_table(self, bits=bits, name=name, description=description):

        mb = Maskbit(bits=bits, name=name, description=description)
        table = mb.to_table(description=True)

        assert isinstance(table, astropy.table.table.Table)
        assert table.colnames == ['bit', 'name', 'description']
        assert table[0]['bit'] == 0
        assert table[0]['name'].decode('utf-8') == 'BITZERO'
        assert table[0]['description'].decode('utf-8') == 'The zeroth bit.'

