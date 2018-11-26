#!/usr/bin/env python
# encoding: utf-8
#
# test_maskbit.py
#
# @Author: Brett Andrews <andrews>
# @Date:   2017-10-06 10:10:00
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-26 12:08:06


from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pytest

from marvin.utils.general.maskbit import Maskbit


bits = [(0, 'BITZERO', 'The zeroth bit.'),
        (1, 'BITONE', 'The first bit.'),
        (2, 'BITTWO', 'The second bit.'),
        (3, 'BITTHREE', 'The third bit'),
        (4, 'BITFOUR', 'The fourth bit')]
schema = pd.DataFrame(bits, columns=['bit', 'label', 'description'])
name = 'MYMASK'
description = 'My first Maskbit.'
mask = np.array([[0, 1], [2, 12]])
custom_mask = np.array([[1, 5], [7, 11]])


class TestMaskbit(object):

    def test_maskbit_init_with_schema(self, name=name, schema=schema, description=description):
        mb = Maskbit(name=name, schema=schema, description=description)
        assert np.all(mb.schema == schema)
        assert mb.name == name
        assert mb.description == description
        assert str(mb) == "<Maskbit 'MYMASK' None>".format(schema)

    @pytest.mark.parametrize('name',
                             ['MANGA_TARGET1',
                              'MANGA_DAPPIXMASK'])
    def test_maskbit_init_from_name(self, name):
        mb = Maskbit(name=name)
        assert mb.name == name
        assert isinstance(mb.schema, pd.DataFrame)
        assert mb.description is None

    def test_values_to_bits_no_value_error(self,  name=name, schema=schema, description=description):
        mb = Maskbit(name=name, schema=schema, description=description)
        with pytest.raises(AssertionError) as ee:
            mb.values_to_bits(values=None)

        assert 'Must provide values.' in str(ee.value)

    @pytest.mark.parametrize('values, expected',
                             [(None, [[[], [0]], [[1], [2, 3]]]),
                              (0, []),
                              (3, [0, 1]),
                              (np.array([1, 3]), [[0], [0, 1]]),
                              (np.array([[0, 2], [1, 5]]), [[[], [1]], [[0], [0, 2]]])])
    def test_values_to_bits(self, values, expected):
        mb = Maskbit(name=name, schema=schema, description=description)
        if values is None:
            mb.mask = mask

        actual = mb.values_to_bits(values=values)
        assert actual == expected

    @pytest.mark.parametrize('value, expected',
                             [(0, []),
                              (3, [0, 1])])
    def test_value_to_bits(self, value, expected):
        mb = Maskbit(name=name, schema=schema, description=description)
        actual = mb._value_to_bits(value, schema.bit.values)
        assert actual == expected

    @pytest.mark.parametrize('values, expected',
                             [(None, [[[], ['BITZERO']], [['BITONE'], ['BITTWO', 'BITTHREE']]]),
                              (0, []),
                              (3, ['BITZERO', 'BITONE']),
                              (np.array([1, 3]), [['BITZERO'], ['BITZERO', 'BITONE']]),
                              (np.array([[0, 2], [1, 3]]), [[[], ['BITONE']], [['BITZERO'], ['BITZERO', 'BITONE']]])])
    def test_values_to_labels(self, values, expected):
        mb = Maskbit(name=name, schema=schema, description=description)
        if values is None:
            mb.mask = mask

        actual = mb.values_to_labels(values=values)
        assert actual == expected

    @pytest.mark.parametrize('bits_in, expected',
                             [([], []),
                              ([0], ['BITZERO']),
                              ([1], ['BITONE']),
                              ([[1]], [['BITONE']]),
                              ([[0, 1], [2]], [['BITZERO', 'BITONE'], ['BITTWO']])])
    def test_bits_to_labels(self, bits_in, expected):
        mb = Maskbit(name=name, schema=schema, description=description)
        actual = mb._bits_to_labels(bits_in)
        assert actual == expected

    @pytest.mark.parametrize('labels, expected',
                             [('BITONE', 2),
                              (['BITONE'], 2),
                              (['BITONE', 'BITTWO'], 6)])
    def test_labels_to_value(self, labels, expected):
        mb = Maskbit(name=name, schema=schema, description=description)
        actual = mb.labels_to_value(labels)
        assert actual == expected

    @pytest.mark.parametrize('labels, expected',
                             [('BITONE', [1]),
                              (['BITONE'], [1]),
                              (['BITONE', 'BITTWO'], [1, 2])])
    def test_labels_to_bits(self, labels, expected):
        mb = Maskbit(name=name, schema=schema, description=description)
        actual = mb.labels_to_bits(labels)
        assert actual == expected

    @pytest.mark.parametrize('labels, expected',
                             [('BITONE', np.array([[0, 0], [2, 0]])),
                              (['BITONE'], np.array([[0, 0], [2, 0]])),
                              (['BITTWO', 'BITTHREE'], np.array([[0, 0], [0, 12]]))])
    def test_get_mask_int(self, labels, expected):
        mb = Maskbit(name=name, schema=schema, description=description)
        mb.mask = mask
        actual = mb.get_mask(labels)
        assert (actual == expected).all()

    @pytest.mark.parametrize('labels, expected',
                             [('BITONE', np.array([[False, False], [True, False]])),
                              (['BITONE'], np.array([[False, False], [True, False]])),
                              (['BITTWO', 'BITTHREE'], np.array([[False, False], [False, True]]))])
    def test_get_mask_bool(self, labels, expected):
        mb = Maskbit(name=name, schema=schema, description=description)
        mb.mask = mask
        actual = mb.get_mask(labels, dtype=bool)
        assert (actual == expected).all()

    @pytest.mark.parametrize('labels, expected',
                             [('BITONE', np.array([[0, 0], [2, 2]])),
                              (['BITONE'], np.array([[0, 0], [2, 2]])),
                              (['BITONE', 'BITTHREE'], np.array([[0, 0], [2, 10]]))])
    def test_get_mask_custom_mask_int(self, labels, expected):
        mb = Maskbit(name=name, schema=schema, description=description)
        mb.mask = mask
        actual = mb.get_mask(labels, mask=custom_mask)
        assert (actual == expected).all()

    @pytest.mark.parametrize('labels, expected',
                             [('BITONE', np.array([[False, False], [True, True]])),
                              (['BITONE'], np.array([[False, False], [True, True]])),
                              (['BITONE', 'BITTHREE'], np.array([[False, False], [True, True]]))])
    def test_get_mask_custom_mask_bool(self, labels, expected):
        mb = Maskbit(name=name, schema=schema, description=description)
        mb.mask = mask
        actual = mb.get_mask(labels, mask=custom_mask, dtype=bool)
        assert (actual == expected).all()

    @pytest.mark.parametrize('labels', ['BITFAKE', ['BITFAKE', 'BITTHREE']])
    def test_get_mask_nonpresent_label(self, labels):

        mb = Maskbit(name=name, schema=schema, description=description)

        with pytest.raises(ValueError) as ee:
            mb.get_mask(labels)

        assert 'label \'BITFAKE\' not found in the maskbit schema.' in str(ee)

    @pytest.mark.parametrize('labels, dtype, expected',
                             [('BITFOUR', bool, np.array([[False, False], [False, False]])),
                              ('BITFOUR', int, np.array([[0, 0], [0, 0]]))])
    def test_get_mask_empty(self, labels, dtype, expected):

        mb = Maskbit(name=name, schema=schema, description=description)
        mb.mask = mask
        actual = mb.get_mask(labels, mask=custom_mask, dtype=dtype)
        assert (actual == expected).all()
