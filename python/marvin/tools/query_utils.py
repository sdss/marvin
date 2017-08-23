#!/usr/bin/env python
# encoding: utf-8

# Licensed under a 3-clause BSD license.

from __future__ import print_function, division
import os
import numpy as np
import six
import yaml
import yamlordereddictloader
from fuzzywuzzy import process
from marvin.utils.dap import datamodel
from marvin import config

# Query Parameter Datamodel
query_params = None


def get_best_fuzzy(name, choices, cutoff=0, return_score=False):
    items = process.extractBests(name, choices, score_cutoff=cutoff)

    if not items:
        return None
    elif len(items) == 1:
        best = items[0]
    else:
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
                best = exact[0]
            else:
                options = [s[0].name if isinstance(s[0], QueryParameter) else s[0] for s in items if s[1] == np.max(scores)]
                raise KeyError('{0} is too ambiguous.  Did you mean one of {1}?'.format(name, options))
        else:
            best = items[0]

    return best if return_score else best[0]


def get_params():
    bestpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', 'query_params_best.cfg')
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

    @property
    def parameters(self):
        ''' List all the queryable parameters '''
        return [param for group in self for param in group]

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
        item = get_best_fuzzy(name, self.list_groups(), cutoff=50)
        if item:
            return self[self.list_groups().index(item)]

    def __contains__(self, name):
        item = get_best_fuzzy(name, self.list_groups(), cutoff=50)
        if item:
            return self[self.list_groups().index(item)]
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
        items (list|dict):
            A list or dictionary of parameters.  If a list of names is input,
            each name will be used as the full name.

    '''
    def __init__(self, name, items):
        self.name = name
        self.score = None

        queryparams = []
        for item in items:
            this_param = self._make_query_parameter(item)
            queryparams.append(this_param)

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

    @property
    def remote(self):
        return self.list_params(remote=True)

    @property
    def short(self):
        return self.list_params(short=True)

    @property
    def display(self):
        return self.list_params(display=True)

    def _make_query_parameter(self, item):
        ''' Create an return a QueryParameter '''
        if isinstance(item, dict):
            item.update({'group': self.name})
            this_param = QueryParameter(**item)
        elif isinstance(item, six.string_types):
            is_best = [p for p in query_params.parameters if item in p.full and (item == p.name or item == p.full)]
            if is_best:
                best_dict = is_best[0].__dict__
                best_dict.update({'group': self.name})
                this_param = QueryParameter(**best_dict)
            else:
                this_param = QueryParameter(item, group=self.name)

        return this_param

    def add_parameter(name):
        pass

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
        item = get_best_fuzzy(name, self.full, cutoff=25)
        if item:
            return self[self.full.index(item)]

    def __contains__(self, name):
        item = get_best_fuzzy(name, self.full, cutoff=25)
        if item:
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
        self.group = kwargs.get('group', None)
        self._set_names()
        self._check_datamodels()

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
        # used for a token string on the web
        self._joinedname = ', '.join([self.full, self.name, self.short, self.display])

    def _check_datamodels(self):
        ''' Check if the query parameter lives in the datamodel '''

        self.property = None
        # DAP datmodels
        dm = datamodel[config.release]
        if self.full in dm:
            self.property = dm[self.full]


bestparams = get_params()
query_params = ParameterGroupList(bestparams)



