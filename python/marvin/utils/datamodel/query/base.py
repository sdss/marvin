# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-22 22:43:15
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-09-28 13:50:08

from __future__ import print_function, division, absolute_import
from marvin.utils.datamodel.query.forms import MarvinForm
from marvin.utils.datamodel import DataModelList
from marvin.core.exceptions import MarvinError
import copy as copy_mod
import os
import numpy as np
import six
import yaml
import yamlordereddictloader
from fuzzywuzzy import process
from marvin.utils.datamodel.dap import datamodel
from marvin import config


__ALL__ = ('QueryDataModelList', 'QueryDataModel')


query_params = None


class QueryDataModel(object):
    """ A class representing a Query datamodel """

    def __init__(self, release, groups=[], aliases=[], exclude=[], **kwargs):

        self.release = release
        self._groups = groups
        self._groups.set_parent(self)
        self.aliases = aliases
        self._exclude = exclude
        self._marvinform = MarvinForm(release=release)
        self._cleanup_keys()

    def __repr__(self):

        return ('<QueryDataModel release={0!r}, n_groups={1}, n_parameters={2}, n_total={3}>'
                .format(self.release, len(self.groups), len(self.parameters), len(self._keys)))

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
        self._keys = newkeys

        # # set parameters
        # self._parameters = [QueryParameter(k) for k in self._keys]

    @property
    def groups(self):
        """Returns the groups for this datamodel. """
        return self._groups

    @groups.setter
    def groups(self, value):
        """Raises an error if trying to set groups directly."""

        raise MarvinError('cannot set groups directly. Use add_groups() instead.')

    def add_group(self, group, copy=True):
        ''' '''
        self._groups.add_group(group, copy=copy)

    def add_groups(self, groups, copy=True):
        ''' '''
        self._groups.add_groups(groups, copy=copy)

    @property
    def parameters(self):
        return QueryList(sum([g.parameters for g in self.groups if g.name.lower() != 'other'], []))

    def add_to_group(self, group, value=None):
        ''' Add free-floating Parameters into a Group '''

        thegroup = self._groups == group
        keys = []
        allkeys = copy_mod.copy(self._keys)
        if value is None:
            allkeys = []
            keys.extend(self._keys)
        else:
            for k in self._keys:
                if value in k:
                    mykey = allkeys.pop(allkeys.index(k))
                    if k not in thegroup.full:
                        keys.append(mykey)
        self._keys = allkeys
        thegroup.add_parameters(keys)

    def regroup(self, values):
        ''' Regroup a set of parameters into groups '''
        for group, value in values.items():
            if isinstance(value, list):
                for v in value:
                    self.add_to_group(group, v)
            else:
                self.add_to_group(group, value)

    @property
    def best(self):
        return [p for p in self.parameters if p.best is True]

    def set_best(self, best):
        ''' sets a list of best query parameters '''
        for b in best:
            if isinstance(b, QueryParameter):
                b.best = True
            elif isinstance(b, six.string_types):
                for i, p in enumerate(self.parameters):
                    if b == p.full:
                        self.parameters[i].best = True

    def use_all_spaxels(self):
        ''' Sets the datamodel to use all the spaxels '''
        self._marvinform = MarvinForm(release=self.release, allspaxels=True)
        self._cleanup_keys()


class QueryDataModelList(DataModelList):
    """A dictionary of Query datamodels."""
    base = {'QueryDataModel': QueryDataModel}


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


class ParameterGroupList(list):
    ''' ParameterGroup Object

    This object inherits from the Python list object. This
    represents a list of query ParameterGroups.

    '''

    def __init__(self, items, best=None):
        self.score = None
        self.best = best
        paramgroups = self._make_groups(items)
        list.__init__(self, paramgroups)

    def _make_groups(self, items):
        if isinstance(items, list):
            paramgroups = [ParameterGroup(item, [], best=self.best) for item in items]
        elif isinstance(items, dict):
            paramgroups = [ParameterGroup(key, vals, best=self.best) for key, vals in items.items()]
        elif isinstance(items, six.string_types):
            paramgroups = ParameterGroup(items, [], best=self.best)
        return paramgroups

    def set_parent(self, parent):
        """Sets parent."""

        assert isinstance(parent, QueryDataModel), 'parent must be a QueryDataModel'
        self.parent = parent

    @property
    def parameters(self):
        ''' List all the queryable parameters '''
        return QueryList([param for group in self for param in group])

    @property
    def groups(self):
        ''' List all the parameter groups '''
        return self.list_groups()

    @groups.setter
    def groups(self, value):
        """Raises an error if trying to set groups directly."""

        raise MarvinError('cannot set groups directly. Use add_groups() instead.')

    def list_groups(self):
        '''Returns a list of query groups.

        Returns:
            names (list):
                A string list of all the Query Group names
        '''
        return [group.name for group in self]

    def add_group(self, group, copy=True):
        ''' '''

        new_grp = copy_mod.copy(group) if copy else group
        if isinstance(new_grp, ParameterGroup):
            self.append(new_grp)
        else:
            new_grp = self._make_groups(new_grp)
            self.append(new_grp)

    def add_groups(self, groups, copy=True):
        ''' '''
        for group in groups:
            self.add_group(group, copy=copy)

    def list_params(self, name_type='full', groups=None):
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
        item = get_best_fuzzy(name, self.groups, cutoff=50)
        if item:
            return self[self.groups.index(item)]

    def __contains__(self, name):
        item = get_best_fuzzy(name, self.groups, cutoff=50)
        if item:
            return True
        else:
            return False

    def __getitem__(self, name):
        if isinstance(name, str):
            return self == name
        else:
            return list.__getitem__(self, name)

    def index(self, value):
        param = self == value
        return self.groups.index(param.name)

    def pop(self, value=None):
        if isinstance(value, int):
            return super(ParameterGroupList, self).pop(value)
        elif not value:
            return super(ParameterGroupList, self).pop()
        else:
            idx = self.index(value)
            return super(ParameterGroupList, self).pop(idx)

    def remove(self, value):
        param = value == self
        tmp = [n for n in self if n != param]
        list.__init__(self, tmp)


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
    def __init__(self, name, items, best=None):
        self.name = name
        self.best = best
        self.score = None

        queryparams = []
        for item in items:
            this_param = self._make_query_parameter(item)
            queryparams.append(this_param)

        list.__init__(self, queryparams)
        self._check_names()

    # def __repr__(self):
    #     old = list.__repr__(self)
    #     old = old.replace('>,', '>,\n')
    #     return ('<ParameterGroup name={0.name}, n_parameters={1}>\n '
    #             '{2}'.format(self, len(self), old))

    def __repr__(self):
        return '<ParameterGroup name={0.name}, n_parameters={1}>'.format(self, len(self))

    def _make_query_parameter(self, item):
        ''' Create and return a QueryParameter '''
        if isinstance(item, dict):
            item.update({'group': self.name, 'best': self.best})
            this_param = QueryParameter(**item)
        elif isinstance(item, six.string_types):
            is_best = [p for p in query_params.parameters if item in p.full and (item == p.name or item == p.full)]
            if is_best:
                best_dict = is_best[0].__dict__
                best_dict.update({'group': self.name, 'best': self.best})
                this_param = QueryParameter(**best_dict)
            else:
                this_param = QueryParameter(item, group=self.name)

        return this_param

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

    @property
    def parameters(self):
        return self.list_params()

    @parameters.setter
    def parameters(self, value):
        """Raises an error if trying to set groups directly."""

        raise MarvinError('cannot set groups directly. Use add_parameters() instead.')

    def add_parameter(self, value, copy=True):
        ''' '''

        new_par = copy_mod.copy(value) if copy else value
        if isinstance(value, QueryParameter):
            self.append(new_par)
        else:
            new_qp = self._make_query_parameter(new_par)
            self.append(new_qp)

    def add_parameters(self, values, copy=True):
        ''' '''
        for value in values:
            self.add_parameter(value, copy=copy)

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
            return QueryList([param for param in paramlist])

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

    def index(self, value):
        param = self == value
        return super(ParameterGroup, self).index(param)

    def pop(self, value=None):
        if isinstance(value, int):
            return super(ParameterGroup, self).pop(value)
        elif not value:
            return super(ParameterGroup, self).pop()
        else:
            idx = self.index(value)
            return super(ParameterGroup, self).pop(idx)

    def remove(self, value):
        param = value == self
        super(ParameterGroup, self).remove(param)

    def _check_names(self):
        names = self.list_params(remote=True)
        for i, name in enumerate(names):
            if names.count(name) > 1:
                self[i].remote = self[i]._under
                self[i].display = '{0} {1}'.format(self[i].table.title(), self[i].display)


class QueryList(list):
    ''' A class for a list of Query Parameters '''

    def __init__(self, items):
        list.__init__(self, items)
        self._full = [s.full for s in self]
        self._remote = [s.remote for s in self]

    def __eq__(self, name):
        item = get_best_fuzzy(name, self, cutoff=25)
        if item:
            return item

    def __contains__(self, name):
        item = get_best_fuzzy(name, self, cutoff=25)
        if item:
            return True
        else:
            return False

    def __getitem__(self, name):
        if isinstance(name, str):
            return self == name
        else:
            return list.__getitem__(self, name)

    def get_full_from_remote(self, value):
        ''' Get the full name from the remote name '''
        return self._full[self._remote.index(value)]


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
        self.schema = kwargs.get('schema', None)
        self.best = kwargs.get('best', None)
        self._set_names()
        self._check_datamodels()

    def __repr__(self):
        return ('<QueryParameter full={0.full}, name={0.name}, short={0.short}, '
                'remote={0.remote}, display={0.display}>'.format(self))

    def _set_names(self):
        ''' Sets alternate names if it can '''
        self._under = self.full.replace('.', '_')
        if not self.table or not self.name:
            if self.full.count('.') == 1:
                self.table, self.name = self.full.split('.')
            elif self.full.count('.') == 2:
                self.schema, self.table, self.name = self.full.split('.')
                self.full = '{0}.{1}'.format(self.table, self.name)
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


# Get the Common Parameters from the filelist

def get_params():
    bestpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data', 'query_params_best.cfg')
    if os.path.isfile(bestpath):
        with open(bestpath, 'r') as stream:
            bestparams = yaml.load(stream, Loader=yamlordereddictloader.Loader)
        return bestparams
    else:
        return None

bestparams = get_params()
query_params = ParameterGroupList(bestparams, best=True)


