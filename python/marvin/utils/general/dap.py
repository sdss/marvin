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


def maps_db2dict_of_props(maps_db_file, x, y):
    """From a ``dapdb.File``, creates a dictionary of properties for the `(x, y)` spaxel."""

    assert isinstance(maps_db_file, marvin.marvindb.dapdb.File)

    dict_of_props = {}

    # Adds the emission lines
    for emline in maps_db_file.emlines:
        value = emline.value[y][x]
        ivar = emline.ivar[y][x] if emline.ivar else 0
        mask = emline.mask[y][x] if emline.mask else 1
        parameter = 'EMLINE_' + emline.parameter.name
        unit = emline.parameter.unit
        channel_name = emline.type.name + '-' + str(emline.type.rest_wavelength)

        if parameter not in dict_of_props:
            dict_of_props[parameter] = {}

        dict_of_props[parameter][channel_name] = {'value': value,
                                                  'ivar': ivar,
                                                  'mask': mask}
        dict_of_props[parameter]['unit'] = unit

    # Idem with the stellar kinematics
    for stellar_kin in maps_db_file.stellarkins:
        value = stellar_kin.value[y][x]
        ivar = stellar_kin.ivar[y][x] if stellar_kin.ivar else 0
        mask = stellar_kin.mask[y][x] if stellar_kin.mask else 1
        parameter = 'STELLAR_' + stellar_kin.parameter.name
        unit = stellar_kin.parameter.unit

        dict_of_props[parameter] = {'NA': {'value': value,
                                           'ivar': ivar,
                                           'mask': mask}}
        dict_of_props[parameter]['unit'] = unit

    # Idem with the spectral indices
    dict_of_props['SPECINDEX'] = {}
    for specindex in maps_db_file.specindices:
        value = specindex.value[y][x]
        ivar = specindex.ivar[y][x] if specindex.ivar else 0
        mask = specindex.mask[y][x] if specindex.mask else 1
        unit = specindex.type.unit
        channel_name = specindex.type.name

        dict_of_props['SPECINDEX'][channel_name] = {'value': value,
                                                    'ivar': ivar,
                                                    'mask': mask}
        dict_of_props['SPECINDEX']['unit'] = unit

    return dict_of_props


def get_dict_of_props_api(plateifu, x, y, dapver=None):
    """Returns a dictionary of DAP properties by performing an API call."""

    url = marvin.config.urlmap['api']['getdap_props']['url']

    try:
        path = 'x={0}/y={1}'.format(x, y)
        url_full = url.format(name=plateifu, path=path)
        response = marvin.api.api.Interaction(url_full, params={'dapver': dapver})
    except Exception as ee:
        raise marvin.core.exceptions.MarvinError(
            'found a problem when checking if remote maps exists: {0}'.format(str(ee)))

    data = response.getData()

    return data


def list_categories(hdu=None, dapver=marvin.config.dapver):
    """Returns a list of categories for a DAP :class:`~marvin.tools.maps.Maps`.

    Parameters:
        hdu (``astropy.io.fits.HDUList`` or None):
            If ``hdu`` is defined, it will use the extension names in the HDU.
            Otherwise it will use the DB.
        dapver (string or None):
            The DAP version to use.

    Returns:
        list_categories (dict):
            A dictionary in which keys are the valid categories an the values
            are the valid channels or None if channels are not appliable for
            that category.

    """

    cat_dict = {}

    if hdu is not None:
        categories = [ext.name.lower() for ext in hdu
                      if ext.name + '_ivar' in hdu and ext.name + '_mask']

        for category in categories:
            extension = hdu[category]
            if len(extension.data.shape) == 2:
                cat_dict[category] = None

            elif len(extension.data.shape) == 3:

                # Gets the channels and creates the names.
                channel_keys = [key for key in extension.header.keys()
                                if re.match('C[0-9]+', key)]
                names = [re.sub('\-+', '-', extension.header[key])
                         for key in channel_keys]

                cat_dict[category] = [name.lower() for name in names]

        return cat_dict

    else:

        mdb = marvin.marvindb

        # Emission lines
        emline_params = ['emline_' + row.name.lower()
                         for row in mdb.session.query(mdb.dapdb.EmLineParameter).all()]
        emline_channels = ['{0}-{1}'.format(row.name.lower(), row.rest_wavelength)
                           for row in mdb.session.query(mdb.dapdb.EmLineType).all()]
        for param in emline_params:
            cat_dict[param] = emline_channels

        # Stellar kinematics
        stell_kin_params = ['stellar_' + row.name.lower()
                            for row in mdb.session.query(mdb.dapdb.StellarKinParameter).all()]
        for param in stell_kin_params:
            cat_dict[param] = None

        # Spectral indices
        specindex_channels = [row.name.lower()
                              for row in mdb.session.query(mdb.dapdb.SpecIndexType).all()]
        cat_dict['specindex'] = specindex_channels


    return cat_dict
