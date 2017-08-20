#!/usr/bin/env python
# encoding: utf-8

# Licensed under a 3-clause BSD license.

from __future__ import print_function, division
import os
import numpy as np
import yaml
import yamlordereddictloader
from fuzzywuzzy import process

# Query Parameter Datamodel


def get_best_fuzzy(name, choices, cutoff=0):
    items = process.extractBests(name, choices, score_cutoff=cutoff)
    if items:
        scores = [s[1] for s in items]
        # finds items with the same score
        morethanone = sum(np.max(scores) == scores) > 1
        if morethanone:
            # tries to find an exact string match
            exact = []
            for s in items:
                itemname = s[0].name if isinstance(s[0], QueryParameter) else s[0]
                if itemname.lower() == name.lower():
                    exact.append(s)
            # returns exact match or fails with ambiguity
            if exact:
                return exact[0]
            else:
                options = [s[0].name if isinstance(s[0], QueryParameter) else s[0] for s in items if s[1] == np.max(scores)]
                raise KeyError('{0} is too ambiguous.  Did you mean one of {1}?'.format(name, options))
        else:
            return items[0]
    else:
        return None


def get_params():
    bestpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data', 'query_params_best.cfg')
    if os.path.isfile(bestpath):
        with open(bestpath, 'r') as stream:
            bestparams = yaml.load(stream, Loader=yamlordereddictloader.Loader)
        return bestparams
    else:
        return None


class ParameterGroupList(list):
    ''' ParameterGroup Object

    This object inherits from the Python list object. This
    represents a list of query ParameterGroups.

    '''

    def __init__(self, items):
        self.score = None
        if isinstance(items, list):
            list.__init__(self, items)
        elif isinstance(items, dict):
            paramgroups = [ParameterGroup(key, vals) for key, vals in items.items()]
            list.__init__(self, paramgroups)

    def list_groups(self):
        '''Returns a list of query groups.

        Returns:
            names (list):
                A string list of all the Query Group names
        '''
        return [group.name for group in self]

    def list_params(self, groups=None, name_type='full'):
        '''Returns a list of parameters from all groups.

        Return a string list of the full parameter names.
        Default is all parameters across all groups.

        Parameters:
            groups (str|list):
                A string or list of strings representing the groups
                of parameters you wish to return
            name_type (str):
                The type of name to generate (full, name, short, remote, display).  Default is full.

        Returns:
            params (list):
                A list of full parameter names
        '''

        assert name_type in ['full', 'short', 'name', 'remote', 'display'], \
            'name_type must be (full, short, name, remote, display)'

        if groups:
            groups = groups if isinstance(groups, list) else [groups]
            grouplist = [self[g] for g in groups]
            return [param.__getattribute__(name_type) for group in grouplist for param in group]
        else:
            return [param.__getattribute__(name_type) for group in self for param in group]

    def __eq__(self, name):
        item = get_best_fuzzy(name, self, cutoff=50)
        if item:
            self.score = item[1]
            return item[0]

    def __contains__(self, name):
        item = get_best_fuzzy(name, self, cutoff=50)
        if item:
            self.score = item[1]
            return item[0]
        else:
            return False

    def __getitem__(self, name):
        if isinstance(name, str):
            return self == name
        else:
            return list.__getitem__(self, name)


class ParameterGroup(list):
    ''' A Query Parameter Group Object

    Query parameters are grouped into specific categories
    for ease of use and navigation.  This object subclasses
    from the Python list object.

    Parameters:
        name (str):
            The name of the group

    '''
    def __init__(self, name, items):
        self.name = name
        self.score = None
        queryparams = [QueryParameter(**item) for item in items]
        list.__init__(self, queryparams)
        self._check_names()

    def __repr__(self):
        old = list.__repr__(self)
        old = old.replace('>,', '>,\n')
        return ('<ParameterGroup name={0.name}, n_parameters={1}>\n '
                '{2}'.format(self, len(self), old))

    @property
    def full(self):
        return self.list_params(full=True)

    def list_params(self, subset=None, display=None, short=None, full=None, remote=None):
        ''' List the parameter names for a given group

        Lists the Query Parameters of the given group

        Parameters:
            subset (str|list):
                String list of a subset of parameters to return
            display (bool):
                Set to return the display names
            short (bool)
                Set to return the short names
            full (bool):
                Set to return the full names
            remote (bool):
                Set to return the remote query names

        Returns:
            param (list):
                The list of parameter
        '''

        if subset:
            params = subset if isinstance(subset, list) else [subset]
            paramlist = [self[g] for g in params]
        else:
            paramlist = self

        if short:
            return [param.short for param in paramlist]
        elif display:
            return [param.display for param in paramlist]
        elif full:
            return [param.full for param in paramlist]
        elif remote:
            return [param.remote for param in paramlist]
        else:
            return [param for param in paramlist]

    def __eq__(self, name):
        item = get_best_fuzzy(name, self, cutoff=25)
        if item:
            self.score = item[1]
            return item[0]

    def __contains__(self, name):
        item = get_best_fuzzy(name, self, cutoff=25)
        if item:
            self.score = item[1]
            return True
        else:
            return False

    def __getitem__(self, name):
        if isinstance(name, str):
            return self == name
        else:
            return list.__getitem__(self, name)

    def _check_names(self):
        names = self.list_params(remote=True)
        for i, name in enumerate(names):
            if names.count(name) > 1:
                self[i].remote = self[i]._under
                self[i].display = '{0} {1}'.format(self[i].table.title(), self[i].display)


class QueryParameter(object):
    ''' A Query Parameter class

    An object representing a query parameter.  Provides access to
    different names for a given parameter.

    Parameters:
        full (str):
            The full naming syntax (table.name) used for all queries.  This name is recommended for full uniqueness.
        table (str):
            The name of the database table the parameter belongs to
        name (str):
            The name of the parameter in the database
        short (str):
            A shorthand name of the parameter
        display (str):
            A display name used for web and plotting purposes.
        dtype (str):
            The type of the parameter (e.g. string, integer, float)
    '''

    def __init__(self, full, table=None, name=None, short=None, display=None,
                 remote=None, dtype=None, **kwargs):
        self.full = full
        self.table = table
        self.name = name
        self.short = short
        self.display = display
        self.dtype = dtype
        self.remote = remote
        self.value = kwargs.get('value', None)
        self._set_names()

    def __repr__(self):
        return ('<QueryParameter full={0.full}, name={0.name}, short={0.short}, '
                'remote={0.remote}, display={0.display}>'.format(self))

    def _set_names(self):
        ''' Sets alternate names if it can '''
        self._under = self.full.replace('.', '_')
        if not self.table or not self.name:
            if '.' in self.full:
                self.table, self.name = self.full.split('.')
            else:
                self.name = self.full
        if not self.short:
            self.short = self.name
        if not self.display:
            self.display = self.name.title()
        if not self.remote:
            self.remote = self._under if 'name' in self.name else self.name

bestparams = get_params()
query_params = ParameterGroupList(bestparams)



