# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-22 22:43:15
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-20 16:47:55

from __future__ import print_function, division, absolute_import
from marvin.utils.datamodel.query.forms import MarvinForm
from marvin.utils.datamodel import DataModelList

__ALL__ = ('QueryDataModelList', 'QueryDataModel')


class QueryDataModel(object):
    """ A class representing a Query datamodel """

    def __init__(self, release, groups=[], aliases=[], exclude=[]):

        self.release = release
        self.groups = groups
        self.aliases = aliases
        self._exclude = exclude
        self._marvinform = MarvinForm(release=release)
        self._cleanup_keys()

    def __repr__(self):

        return ('<QueryDataModel release={0!r}, n_groups={1}, n_parameters={2}>'
                .format(self.release, len(self.groups), len(self.keys)))

    def _cleanup_keys(self):
        ''' Cleans up the list for MarvinForm keys '''

        # get all the keys in the marvin form
        keys = list(self._marvinform._param_form_lookup.keys())
        keys.sort()

        # simplify the spaxelprop list down to one set
        mykeys = [k.split('.', 1)[-1] for k in keys if 'cleanspaxel' not in k]
        mykeys = [k.replace(k.split('.')[0], 'spaxelprop') if 'spaxelprop'
                  in k else k for k in mykeys]

        # replace table names with shortcut names
        rev = {v: k for k, v in self._marvinform._param_form_lookup._tableShortcuts.items()}
        newkeys = [k.replace(k.split('.')[0], rev[k.split('.')[0]]) if k.split('.')[0] in rev.keys() else k for k in mykeys]

        # exclude tables from list of keys
        if self._exclude:
            for table in self._exclude:
                newkeys = [k for k in newkeys if table not in k]

        # final sort and set
        newkeys.sort()
        self.keys = newkeys


class QueryDataModelList(DataModelList):
    """A dictionary of Query datamodels."""
    base = {'QueryDataModel': QueryDataModel}




