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

import pytest
import pandas as pd
import numpy as np

from marvin.utils.dap.datamodel import Maskbit
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

bits = [(0, 'BITZERO', 'The zeroth bit.'),
        (1, 'BITONE', 'The first bit.'),
        (2, 'BITTWO', 'The second bit.'),
        (3, 'BITTHREE', 'The third bit')]
schema = pd.DataFrame(bits, columns=['bit', 'label', 'description'])
name = 'MYMASK'
description = 'My first Maskbit.'
mask = np.array([[0, 1], [2, 12]])


class TestMaskbit(object):
    
    def test_maskbit_init(self, schema=schema, name=name, description=description):
        mb = Maskbit(schema=schema, name=name, description=description)
        assert np.all(mb.schema == schema)
        assert mb.name == name
        assert mb.description == description
        assert str(mb) == "<Maskbit 'MYMASK'\n\n{0!r}>".format(schema)

    def test_values_to_bits_no_value_error(self, schema=schema, name=name, description=description):
        mb = Maskbit(schema=schema, name=name, description=description)
        with pytest.raises(AssertionError) as ee:
            mb.values_to_bits(value=None)

        assert 'Must provide a value.' in str(ee.value)

    @pytest.mark.parametrize('value, expected',
                             [(None, [[[], [0]], [[1], [2, 3]]]),
                              (0, []),
                              (3, [0, 1]),
                              (np.array([1, 3]), [[0], [0, 1]]),
                              (np.array([[0, 2], [1, 5]]), [[[], [1]], [[0], [0, 2]]])])
    def test_values_to_bits(self, value, expected):
        mb = Maskbit(schema=schema, name=name, description=description)
        if value is None:
            mb.mask = mask
        actual = mb.values_to_bits(value=value)
        assert actual == expected

    @pytest.mark.parametrize('value, expected',
                             [(None, [[[], ['BITZERO']], [['BITONE'], ['BITTWO', 'BITTHREE']]]),
                              (0, []),
                              (3, ['BITZERO', 'BITONE']),
                              (np.array([1, 3]), [['BITZERO'], ['BITZERO', 'BITONE']]),
                              (np.array([[0, 2], [1, 3]]), [[[], ['BITONE']], [['BITZERO'], ['BITZERO', 'BITONE']]])])
    def test_values_to_labels(self, value, expected):
        mb = Maskbit(schema=schema, name=name, description=description)
        if value is None:
            mb.mask = mask
        actual = mb.values_to_labels(value=value)
        assert actual == expected

    @pytest.mark.parametrize('bits_in, expected',
                             [([], []),
                              ([0], ['BITZERO']),
                              ([1], ['BITONE']),
                              ([[1]], [['BITONE']]),
                              ([[0, 1], [2]], [['BITZERO', 'BITONE'], ['BITTWO']])])
    def test_bits_to_labels(self, bits_in, expected):
        mb = Maskbit(schema=schema, name=name, description=description)
        actual = mb._bits_to_labels(bits_in)
        assert actual == expected

    @pytest.mark.parametrize('labels, expected',
                             [('BITONE', 2),
                              (['BITONE'], 2),
                              (['BITONE', 'BITTWO'], 6)])
    def test_labels_to_value(self, labels, expected):
        mb = Maskbit(schema=schema, name=name, description=description)
        mb.mask = mask
        actual = mb.labels_to_value(labels)
        assert actual == expected