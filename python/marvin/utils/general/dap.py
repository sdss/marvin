#!/usr/bin/env python
# encoding: utf-8
#
# dap.py
#
# Created by José Sánchez-Gallego on 15 Jun 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re

import astropy.io.fits as fits

import marvin


def maps_file2dict_of_props(maps_file, x, y):
    """From a MAPS file, creates a dictionary of properties for the `(x, y)` spaxel."""

    if not isinstance(maps_file, fits.HDUList):
        fits_map = fits.open(maps_file)
    else:
        fits_map = maps_file

    dict_of_props = {}

    # Selects extensions that contains a "category", i.e., if it has a value,
    # a mask and an ivar.
    valid_extensions = [extension.name for extension in fits_map
                        if extension.name + '_ivar' in fits_map and extension.name + '_mask']

    for ext_name in valid_extensions:
        category = ext_name
        extension = fits_map[ext_name]
        ivar_extension = fits_map[ext_name + '_ivar']
        mask_extension = fits_map[ext_name + '_mask']

        # Checks whether we have channels for this category
        if len(extension.data.shape) == 2:

            dict_of_props[category] = {
                'NA': {'value': extension.data[y, x],
                       'ivar': ivar_extension.data[y, x],
                       'mask': mask_extension.data[y, x]},
                'unit': extension.header['BUNIT'] if 'BUNIT' in extension.header else ''}

        elif len(extension.data.shape) == 3:

            # Gets the channels and creates the names.
            channel_keys = [key for key in extension.header.keys() if re.match('C[0-9]+', key)]
            names = [re.sub('\-+', '-', extension.header[key]) for key in channel_keys]

            dict_of_props[category] = {
                name: {'value': extension.data[ii, y, x],
                       'ivar': ivar_extension.data[ii, y, x],
                       'mask': mask_extension.data[ii, y, x]}
                for ii, name in enumerate(names)}
            dict_of_props[category]['unit'] = (extension.header['BUNIT']
                                               if 'BUNIT' in extension.header else '')

        else:
            raise ValueError('error when obtaining properties from '
                             'extension {0}: it has {1} dimensions!'
                             .format(ext_name, len(extension.data.shape)))

    return dict_of_props
