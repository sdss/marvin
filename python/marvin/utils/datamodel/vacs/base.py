# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2018-07-17 23:36:31
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-19 00:37:28

from __future__ import print_function, division, absolute_import

import copy as copy_mod
import os

import astropy.table as table
from astropy import units as u

from marvin.core.exceptions import MarvinError
from marvin.contrib.vacs.base import VACMixIn
from marvin.utils.datamodel import DataModelList
from marvin.utils.general.structs import FuzzyList


class VACDataModel(object):
    """A class representing a VAC datamodel """

    def __init__(self, release, aliases=[], vacs=[]):

        self.release = release
        self.aliases = aliases

        self.vacs = VACList(vacs, parent=self)

    def __repr__(self):

        return ('<VACDataModel release={0!r}, n_vacs={1}>'
                .format(self.release, len(self.vacs)))

    def copy(self):
        """Returns a copy of the datamodel."""

        return copy_mod.deepcopy(self)

    def __eq__(self, value):
        """Uses fuzzywuzzy to return the closest property match."""

        vac_names = [vac.name for vac in self.vacs]

        if value in vac_names:
            return self.vacs[vac_names.index(value)]

        try:
            vac_best_match = self.vacs[value]
        except ValueError:
            vac_best_match = None

        if vac_best_match is None:
            raise ValueError('too ambiguous input {!r}'.format(value))
        elif vac_best_match is not None:
            return vac_best_match

    def __contains__(self, value):

        try:
            match = self.__eq__(value)
            if match is None:
                return False
            else:
                return True
        except ValueError:
            return False

    def __getitem__(self, value):
        return self == value


class VACDataModelList(DataModelList):
    """A dictionary of DRP datamodels."""

    base = {'VACDataModel': VACDataModel}


class VACList(FuzzyList):
    ''' A fuzzy list of available VACs '''

    def __init__(self, the_list, parent=None):

        self.parent = parent

        super(VACList, self).__init__([])

        for item in the_list:
            self.append(item, copy=True)

    def mapper(self, value):
        """Helper method for the fuzzy list to match on the vac name."""

        return value.name

    def append(self, value, copy=True):
        """Appends with copy."""

        append_obj = value if copy is False else copy_mod.deepcopy(value)
        append_obj.parent = self.parent

        if issubclass(append_obj, VACMixIn):
            super(VACList, self).append(append_obj)
        else:
            raise ValueError('invalid vac of type {!r}'.format(type(append_obj)))

    def list_names(self):
        """Returns a list with the names of the vacs in this list."""

        return [item.name for item in self]

    def to_table(self, pprint=False, description=True, max_width=1000):
        """Returns an astropy table with all the vacs in this datamodel.

        Parameters:
            pprint (bool):
                Whether the table should be printed to screen using astropy's
                table pretty print.
            description (bool):
                If ``True``, an extra column with the description of the
                vac will be added.
            max_width (int or None):
                A keyword to pass to ``astropy.table.Table.pprint()`` with the
                maximum width of the table, in characters.

        Returns:
            result (``astropy.table.Table``):
                If ``pprint=False``, returns an astropy table containing
                the name of the vac, whether it has ``ivar`` or
                ``mask``, the units, and a description (if
                ``description=True``)..

        """

        vac_table = table.Table(
            None, names=['name', 'version', 'description'],
            dtype=['S20', 'S20', 'S500'])

        if self.parent:
            vac_table.meta['release'] = self.parent.release

        for vac in self:
            vac_table.add_row((vac.name,
                               vac.version[self.parent.release],
                               vac.description))

        if not description:
            vac_table.remove_column('description')

        if pprint:
            vac_table.pprint(max_width=max_width, max_lines=1e6)
            return

        return vac_table

    def write_csv(self, filename=None, path=None, overwrite=None, **kwargs):
        ''' Write the datamodel to a CSV '''

        release = self.parent.release.lower().replace('-', '')

        if not filename:
            filename = 'vacs_dm_{0}.csv'.format(release)

        if not path:
            path = os.path.join(os.getenv("MARVIN_DIR"), 'docs', 'sphinx', '_static')

        fullpath = os.path.join(path, filename)
        table = self.to_table(**kwargs)
        table.write(fullpath, format='csv', overwrite=overwrite)

