#!/usr/bin/env python
# encoding: utf-8
#
# maskbit.py
#
# @Author: Brett Andrews <andrews>
# @Date:   2017-10-06 10:10:00
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-26 11:51:50

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd

import marvin
from marvin.utils.yanny import yanny


# Stores the maskbits yanny file structure so that we don't need to open it more than once.
_maskbits_from_yanny = None


def _read_maskbit_schemas():
    """Read all available SDSS maskbit schemas from yanny file.

    Returns:
        Record Array: all bits for all schemas.
    """

    global _maskbits_from_yanny

    if _maskbits_from_yanny is None:
        path_maskbits = os.path.join(os.path.dirname(marvin.__file__), 'data', 'sdssMaskbits.par')
        _maskbits_from_yanny = yanny(path_maskbits, np=True)

    return _maskbits_from_yanny['MASKBITS']


def get_available_maskbits():
    """Get names of available maskbit schemas from yanny file.

    Returns:
        list: Names of available maskbits.
    """
    maskbits = _read_maskbit_schemas()
    return sorted(set([it[0] for it in maskbits]))


def get_manga_target(flag_id, bitmasks, header):
    """Get MANGA_TARGET[``flag_id``] flag.

    Parameters:
        flag_id (str):
            Flag ID number (e.g., "1" for MANGA_TARGET1).
        bitmasks (dict):
            `Maskbit` objects.
        header (`astropy.io.fits.header.Header`):
            File header.

    Returns:
        `Maskbit`
    """
    flag_id = str(int(flag_id))
    manga_target = bitmasks['MANGA_TARGET{}'.format(flag_id)]

    try:
        manga_target.mask = int(header['MNGTRG{}'.format(flag_id)])
    except KeyError:
        manga_target.mask = int(header['MNGTARG{}'.format(flag_id)])

    return manga_target


class Maskbit(object):
    """A class representing a maskbit.

    Parameters:
        schema (DataFrame):
            Maskbit schema.
        name (str):
            Name of maskbit.
        description (str):
            Description of maskbit.
    """

    def __init__(self, name, schema=None, description=None):

        self.name = name
        self.schema = schema if schema is not None else self._load_schema(name)
        self.description = description if description is not None else None
        self.mask = None

    def __repr__(self):
        if (isinstance(self.mask, int) or self.mask is None):
            labels = self.labels
        else:
            labels = 'shape={}'.format(self.mask.shape)
        return '<Maskbit {0!r} {1}>'.format(self.name, labels)

    def _load_schema(self, flag_name):
        """Load SDSS Maskbit schema from yanny file.

        Parameters:
            flag_name (str):
                Name of flag.

        Returns:
            DataFrame: Schema of flag.
        """
        maskbits = _read_maskbit_schemas()
        flag = maskbits[maskbits['flag'] == flag_name]

        return pd.DataFrame(flag[['bit', 'label', 'description']])

    @property
    def bits(self):
        return self.values_to_bits() if self.mask is not None else None

    @property
    def labels(self):
        return self.values_to_labels() if self.mask is not None else None

    def values_to_bits(self, values=None):
        """Convert mask values to a list of bits set.

        Parameters:
            values (int or array):
                Mask values. If ``None``, apply to entire
                ``Maskbit.mask`` array.  Default is ``None``.

        Returns:
            list:
                Bits that are set.

        Example:
            >>> maps = Maps(plateifu='8485-1901')
            >>> ha = maps['emline_gflux_ha_6564']
            >>> ha.pixmask.values_to_bits()
            [[[0, 1, 4, 30],
              [0, 1, 4, 30],
              ...
              [0, 1, 4, 30]]]
        """
        # assert (self.mask is not None) or (values is not None), 'Must provide values.'

        # values = np.array(self.mask) if values is None else np.array(values)
        # ndim = values.ndim

        # assert ndim <= 3, '`value` must be int, 1-D array, 2-D array, or 3-D array.'

        # # expand up to 2 dimensions
        # while values.ndim < 3:
        #     values = np.array([values])

        # # create list of list of lists of bits set
        # bits_set = []
        # for ii in range(values.shape[0]):
        #     row_ii = []
        #     for jj in range(values.shape[1]):
        #         row_jj = []
        #         for kk in range(values.shape[2]):
        #             row_jj.append(self._value_to_bits(values[ii, jj, kk], self.schema.bit.values))
        #         row_ii.append(row_jj)
        #     bits_set.append(row_ii)

        # # condense back down to initial dimensions
        # for __ in range(3 - ndim):
        #     bits_set = bits_set[0]

        bits_set = self._get_a_set(values, convert_to='bits')

        return bits_set

    def _get_uniq_bits(self, values):
        ''' Return a dictionary of unique bits

        Parameters:
            values (list):
                A flattened list of mask values

        Returns:
            dict:
                A unique dictionary of {mask value: bit list} as {key: value}
        '''
        uniqvals = set(values)
        vdict = {v: self._value_to_bits(v, self.schema.bit.values) for v in uniqvals}
        return vdict

    def _get_uniq_labels(self, values):
        ''' Return a dictionary of unique labels

        Parameters:
            values (list):
                A flattened list of mask values

        Returns:
            dict:
                A unique dictionary of {mask value: labels list} as {key: value}
        '''
        uniqbits = self._get_uniq_bits(values)
        uniqlabels = {k: self.schema.label[self.schema.bit.isin(v)].values.tolist() for k, v in uniqbits.items()}
        return uniqlabels

    def _get_a_set(self, values, convert_to='bits'):
        ''' Convert mask values to a list of either bit or label sets.

        Parameters:
            values (int or array):
                Mask values. If ``None``, apply to entire
                ``Maskbit.mask`` array.  Default is ``None``.
            convert_to (str):
                Indicates what to convert to.  Either "bits" or "labels"

        Returns:
            list:
                Bits/Labels that are set.

        '''
        assert (self.mask is not None) or (values is not None), 'Must provide values.'

        values = np.array(self.mask) if values is None else np.array(values)
        ndim = values.ndim
        shape = values.shape

        assert ndim <= 3, '`value` must be int, 1-D array, 2-D array, or 3-D array.'

        flatmask = values.flatten()

        if convert_to == 'bits':
            uniqvals = self._get_uniq_bits(flatmask)
        elif convert_to == 'labels':
            uniqvals = self._get_uniq_labels(flatmask)

        vallist = list(map(lambda x: uniqvals[x], flatmask))
        if ndim > 0:
            vals_set = np.array(vallist, dtype=object).reshape(shape).tolist()
        else:
            vals_set = vallist[0]

        return vals_set

    def _value_to_bits(self, value, bits_all):
        """Convert mask value to a list of bits.

        Parameters:
            value (int):
                Mask value.
            bits_all (array):
                All bits for flag.

        Returns:
            list:
                Bits that are set.
        """
        return [it for it in bits_all if int(value) & (1 << it)]

    def values_to_labels(self, values=None):
        """Convert mask values to a list of the labels of bits set.

        Parameters:
            values (int or array):
                Mask values. If ``None``, apply to entire
                ``Maskbit.mask`` array.  Default is ``None``.

        Returns:
            list:
                Bits that are set.

        Example:
            >>> maps = Maps(plateifu='8485-1901')
            >>> ha = maps['emline_gflux_ha_6564']
            >>> ha.pixmask.values_to_labels()
            [[['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE'],
              ['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE'],
               ...
              ['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE']]]
        """
        #bits_set = self.values_to_bits(values=values)
        #labels_set = self._bits_to_labels(bits_set)

        labels_set = self._get_a_set(values, convert_to='labels')

        return labels_set

    def _bits_to_labels(self, nested):
        """Recursively convert a nested list of bits to labels.

        Parameters:
            nested (list):
                Nested list of bits.

        Returns:
            list: Nested list of labels.
        """
        # Base condition
        if isinstance(nested, (int, np.integer)):
            return self.schema.label[self.schema.bit == nested].values[0]

        return [self._bits_to_labels(it) for it in nested]

    def labels_to_value(self, labels):
        """Convert bit labels into a bit value.

        Parameters:
            labels (str or list):
                Labels of bits to set.

        Returns:
            int: Integer bit value.

        Example:
            >>> maps = Maps(plateifu='8485-1901')
            >>> ha = maps['emline_gflux_ha_6564']
            >>> ha.pixmask._labels_to_value('DONOTUSE')
            1073741824

            >>> ha.pixmask._labels_to_value(['NOCOV', 'LOWCOV'])
            3
        """
        if isinstance(labels, str):
            labels = [labels]

        bit_values = []
        for label in labels:
            bit = self.schema.bit[self.schema.label == label]
            if not bit.empty:
                bit_values.append(bit.values[0])

        return np.sum([2**value for value in bit_values])

    def labels_to_bits(self, labels):
        """Convert bit labels into bits.

        Parameters:
            labels (str or list):
                Labels of bits.

        Returns:
            list: Bits that correspond to the labels.

        Example:
            >>> maps = Maps(plateifu='8485-1901')
            >>> ha = maps['emline_gflux_ha_6564']
            >>> ha.pixmask.labels_to_bits('DONOTUSE')
            [30]

            >>> ha.pixmask.labels_to_value(['NOCOV', 'LOWCOV'])
            [0, 1]
        """
        return self.values_to_bits(self.labels_to_value(labels))

    def get_mask(self, labels, mask=None, dtype=int):
        """Create mask from a list of labels.

        If ``dtype`` is ``int``, then ``get_mask`` can effectively
        perform an OR or AND operation.  However, if ``dtype`` is
        ``bool``, then ``get_mask`` does an OR.

        Parameters:
            labels (str or list):
                Labels of bits.
            mask (int or array):
                User-defined mask. If ``None``, use ``self.mask``.
                Default is ``None``.
            dtype:
                Output dtype, which must be either ``int`` or ``bool``.
                Default is ``int``.

        Returns:
            array: Mask for given labels.

        Example:
            >>> maps = Maps(plateifu='8485-1901')
            >>> ha = maps['emline_gflux_ha_6564']
            >>> ha.pixmask.get_mask(['NOCOV', 'LOWCOV'])
            array([[3, 3, 3, ..., 3, 3, 3],
                   ...,
                   [3, 3, 3, ..., 3, 3, 3]])

            >>> ha.pixmask.get_mask(['NOCOV', 'LOWCOV'], dtype=bool)
            array([[ True,  True,  True, ...,  True,  True,  True],
                   ...,
                   [ True,  True,  True, ...,  True,  True,  True]], dtype=bool)
        """

        assert dtype in [int, bool], '``dtype`` must be either ``int`` or ``bool``.'

        if isinstance(labels, str):
            labels = [labels]

        schema_labels = self.schema.label.tolist()
        for label in labels:
            if label not in schema_labels:
                raise ValueError('label {0!r} not found in the maskbit schema.'.format(label))

        bits = self.labels_to_bits(labels)
        mask = mask if mask is not None else self.mask

        if len(bits) == 0:
            return np.zeros(mask.shape, dtype=int)

        return np.sum([mask & 2**bit for bit in bits], axis=0).astype(dtype)
