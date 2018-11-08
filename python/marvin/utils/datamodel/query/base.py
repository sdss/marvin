# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-08-22 22:43:15
# @Last modified by:   Brian Cherinka
# @Last modified time: 2018-11-08 12:26:08

from __future__ import absolute_import, division, print_function

import copy as copy_mod
import inspect
import os
import warnings

import numpy as np
import six
import yaml
import yamlordereddictloader
from astropy.table import Table
from fuzzywuzzy import fuzz, process
from sqlalchemy_utils import get_hybrid_properties

from marvin import config
from marvin.core.exceptions import MarvinError, MarvinUserWarning
from marvin.utils.datamodel import DataModelList
from marvin.utils.datamodel.maskbit import get_maskbits
from marvin.utils.general.structs import FuzzyList


if config.db:
    from marvin.utils.datamodel.query.forms import MarvinForm
else:
    MarvinForm = None


__ALL__ = ('QueryDataModelList', 'QueryDataModel')


query_params = None

PARAM_CACHE = {}


class QueryDataModel(object):
    """ A class representing a Query datamodel """

    def __init__(self, release, groups=[], aliases=[], exclude=[], **kwargs):

        self.release = release
        self._groups = groups
        self._groups.set_parent(self)
        self.aliases = aliases
        self._exclude = exclude
        self.dap_datamodel = kwargs.get('dapdm', None)
        self.bitmasks = get_maskbits(self.release)
        self._mode = kwargs.get('mode', config.mode)
        self._get_parameters()
        self._check_datamodels()

    def __repr__(self):

        return ('<QueryDataModel release={0!r}, n_groups={1}, n_parameters={2}, n_total={3}>'
                .format(self.release, len(self.groups), len(self.parameters), len(self._keys)))

    def copy(self):
        """Returns a copy of the datamodel."""

        return copy_mod.deepcopy(self)

    def _get_parameters(self):
        ''' Get the parameters for the datamodel '''

        # check cache for parameters
        if self.release in PARAM_CACHE:
            self._keys = PARAM_CACHE[self.release]
            return

        # get the parameters
        if self._mode == 'local':
            self._marvinform = MarvinForm(release=self.release)
            self._cleanup_keys()
        elif self._mode == 'remote':
            self._get_from_remote()
        elif self._mode == 'auto':
            if config.db:
                self._mode = 'local'
            else:
                self._mode = 'remote'
            self._get_parameters()

    def _get_from_remote(self):
        ''' Get the keys from a remote source '''

        from marvin.api.api import Interaction

        # if not urlmap then exit
        if not config.urlmap:
            self._keys = []
            return

        # try to get the url
        try:
            url = config.urlmap['api']['getallparams']['url']
        except Exception as e:
            warnings.warn('Cannot access Marvin API to get the full list of query parameters. '
                          'for the Query Datamodel. Only showing the best ones.', MarvinUserWarning)
            url = None
            self._keys = []
            #self._cleanup_keys()

        # make the call
        if url:
            try:
                ii = Interaction(url, params={'release': self.release, 'paramdisplay': 'all'})
            except Exception as e:
                warnings.warn('Could not remotely retrieve full set of parameters. {0}'.format(e), MarvinUserWarning)
                self._keys = []
            else:
                # this deals with all parameters from all releases at once
                PARAM_CACHE.update(ii.getData())
                self._check_aliases()

                for key in list(PARAM_CACHE.keys()):
                    self._keys = PARAM_CACHE[key] if key in PARAM_CACHE else []
                    self._remove_query_params()

    def _check_aliases(self):
        ''' Check the release of the return parameters against the aliases '''
        for key, val in PARAM_CACHE.items():
            if key != self.release and key in self.aliases:
                PARAM_CACHE[self.release] = val

    def _remove_query_params(self):
        ''' Remove keys from query_params best list '''
        origlist = query_params.list_params('full')
        for okey in origlist:
            if okey in self._keys:
                self._keys.remove(okey)

        # add the final list to the cache
        if self.release not in PARAM_CACHE:
            PARAM_CACHE[self.release] = self._keys

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
        newkeys = [k.replace(k.split('.')[0],
                             rev[k.split('.')[0]]) if k.split('.')[0] in rev.keys()
                   else k for k in mykeys]

        # remove any hidden keys
        newkeys = [n for n in newkeys if '._' not in n]

        # exclude tables from list of keys
        if self._exclude:
            for table in self._exclude:
                newkeys = [k for k in newkeys if table not in k]

        # final sort and set
        newkeys.sort()
        self._keys = newkeys

        # remove keys from query_params best list here
        self._remove_query_params()

    def _check_datamodels(self, parameters=None):
        ''' Check and match the datamodels '''

        params = [parameters] if parameters else self.parameters

        # DAP datamodel
        for qp in params:
            if self.dap_datamodel:
                try:
                    qp.property = self.dap_datamodel[qp.full]
                except ValueError as e:
                    pass

    def __eq__(self, value):
        """Uses fuzzywuzzy to return the closest parameter/group match."""

        # Gets the best match for parameters and groups. If there is a match
        # in parameters, returns it. Otherwise tries groups.

        try:
            param_best_match = self.parameters[value]
            if param_best_match:
                return param_best_match
        except (KeyError, ValueError):
            pass

        try:
            group_best_match = self.groups[value]
            if group_best_match:
                return group_best_match
        except (KeyError, ValueError):
            pass

        raise ValueError('too ambiguous input {!r}'.format(value))

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
        self._groups.add_group(group, copy=copy, parent=self)

    def add_groups(self, groups, copy=True):
        ''' '''
        self._groups.add_groups(groups, copy=copy, parent=self)

    @property
    def parameters(self):
        return QueryList(sum([g.parameters for g in self.groups], []))

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
        return QueryList([p for p in self.parameters if p.best is True])

    @property
    def best_groups(self):
        return self.groups.best

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

    def _reset_marvinform(self):
        self._marvinform = MarvinForm(release=self.release)

    def to_table(self, pprint=False, max_width=1000, only_best=False, db=False):
        ''' Write the datamodel to an Astropy table '''

        param_table = Table(None, names=['group', 'full', 'best', 'name', 'display',
                                         'db_schema', 'db_table', 'db_column', 'is_hybrid'],
                            dtype=['S20', 'S300', bool, 'S30', 'S30', 'S30', 'S30', 'S30', bool])

        iterable = self.groups.parameters

        param_table.meta['release'] = self.release

        for param in iterable:
                param_table.add_row((param.group, param.full, param.best,
                                     param.name, param.display,
                                     param.db_schema, param.db_table,
                                     param.db_column, param.is_hybrid()))

        if only_best:
            notbest = param_table['best'] is False
            param_table.remove_rows(notbest)

        if not db:
            param_table.remove_columns(['db_schema', 'db_table', 'db_column', 'is_hybrid'])

        if pprint:
            param_table.pprint(max_width=max_width, max_lines=1e6)
            return

        return param_table

    def write_csv(self, filename=None, path=None, overwrite=None, **kwargs):
        ''' Write the datamodel to a CSV '''

        release = self.release.lower().replace('-', '')

        if not filename:
            filename = 'query_dm_{0}.csv'.format(release)

        if not path:
            path = os.path.join(os.getenv("MARVIN_DIR"), 'docs', 'sphinx', '_static')

        fullpath = os.path.join(path, filename)
        table = self.to_table(**kwargs)
        table.write(fullpath, format='csv', overwrite=overwrite)


class QueryDataModelList(DataModelList):
    """A dictionary of Query datamodels."""
    base = {'QueryDataModel': QueryDataModel}


def get_best_fuzzy(name, choices, cutoff=60, return_score=False):
    items = process.extractBests(name, choices, score_cutoff=cutoff, scorer=fuzz.WRatio)

    if not items:
        best = None
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
                options = [s[0].name if isinstance(s[0], QueryParameter)
                           else s[0] for s in items if s[1] == np.max(scores)]
                raise KeyError('{0} is too ambiguous.  '
                               'Did you mean one of {1}?'.format(name, options))
        else:
            best = items[0]

    if best is None:
        raise ValueError('Could not find a match for {0}.  Please refine your text.'.format(name))

    return best if return_score else best[0]


def strip_mapped(self):
    ''' Strip the mapped items for display with dir

    Since __dir__ cannot have . in the attribute name, this
    strips the returned mapper(item) parameter of any . in the name.
    Used for query parameter syntax [table.parameter_name]

    For cases where the parameter_name is "name", and thus
    non-unique, it also returns the mapper name with "." replaced
    with "_", to make unique. "ifu.name" becomes "ifu_name", etc.

    Parameters:
        self:
            a QueryFuzzyList object

    Returns:
        list of mapped named stripped of dots

    '''
    params = []
    for item in self:
        mapped_item = self.mapper(item)
        if '.' in mapped_item:
            parta, name = mapped_item.split('.')
            if name == 'name':
                name = mapped_item.replace('.', '_')
            params.append(name)
        else:
            params.append(mapped_item)
    return params


class QueryFuzzyList(FuzzyList):
    ''' Fuzzy List for Query Parameters '''

    def __dir__(self):
        class_members = list(list(zip(*inspect.getmembers(self.__class__)))[0])
        params = strip_mapped(self)
        return class_members + params

    def __getattr__(self, value):

        mapped_values = [super(QueryFuzzyList, self).__getattribute__('mapper')(item)
                         for item in self]
        stripped_values = strip_mapped(self)

        if value in stripped_values:
            return self[mapped_values[stripped_values.index(value)]]

        return super(QueryFuzzyList, self).__getattribute__(value)

    def index(self, value):
        param = self == value
        index = [i for i, item in enumerate(self) if item is param]

        if index:
            return index[0]
        else:
            raise ValueError('{0} is not in the list'.format(value))

    def pop(self, value=None):
        if isinstance(value, int):
            return super(QueryFuzzyList, self).pop(value)
        elif not value:
            return super(QueryFuzzyList, self).pop()
        else:
            idx = self.index(value)
            return super(QueryFuzzyList, self).pop(idx)

    def remove(self, value):
        param = value == self
        tmp = [n for n in self if n != param]
        QueryFuzzyList.__init__(self, tmp, use_fuzzy=get_best_fuzzy)


class ParameterGroupList(QueryFuzzyList):
    ''' ParameterGroup Object

    This object inherits from the Python list object. This
    represents a list of query ParameterGroups.

    '''

    def __init__(self, items):
        self.score = None
        paramgroups = self._make_groups(items)
        QueryFuzzyList.__init__(self, paramgroups, use_fuzzy=get_best_fuzzy)

    def mapper(self, value):
        return value.name.lower().replace(' ', '_').replace('.', '_')

    def _make_groups(self, items, best=None):
        if isinstance(items, list):
            paramgroups = [ParameterGroup(item, []) for item in items]
        elif isinstance(items, dict):
            paramgroups = [ParameterGroup(key, vals)
                           for key, vals in items.items()]
        elif isinstance(items, six.string_types):
            paramgroups = ParameterGroup(items, [])
        return paramgroups

    def set_parent(self, parent):
        """Sets parent."""

        assert isinstance(parent, QueryDataModel), 'parent must be a QueryDataModel'
        self.parent = parent
        for item in self:
            item.set_parent(parent)

    @property
    def parameters(self):
        ''' List all the queryable parameters '''
        return QueryList([param for group in self for param in group])

    @property
    def names(self):
        ''' List all the parameter groups '''
        return self.list_groups()

    @property
    def best(self):
        ''' List the best parameters in each group '''
        grp_copy = copy_mod.deepcopy(self)
        grp_copy.__init__(bestparams)
        grp_copy.parent._check_datamodels()
        return grp_copy

    @names.setter
    def names(self, value):
        """Raises an error if trying to set groups directly."""

        raise MarvinError('cannot set names directly. Use add_groups() instead.')

    def list_groups(self):
        '''Returns a list of query groups.

        Returns:
            names (list):
                A string list of all the Query Group names
        '''
        return [group.name for group in self]

    def add_group(self, group, copy=True, parent=None):
        ''' '''

        new_grp = copy_mod.copy(group) if copy else group
        if isinstance(new_grp, ParameterGroup):
            self.append(new_grp)
        else:
            new_grp = self._make_groups(new_grp)
            new_grp.set_parent(parent)
            self.append(new_grp)

    def add_groups(self, groups, copy=True, parent=None):
        ''' '''
        for group in groups:
            self.add_group(group, copy=copy, parent=parent)

    def list_params(self, name_type='full', groups=None):
        '''Returns a list of parameters from all groups.

        Return a string list of the full parameter names.
        Default is all parameters across all groups.

        Parameters:
            groups (str|list):
                A string or list of strings representing the groups
                of parameters you wish to return
            name_type (str):
                The type of name to generate (full, name, short, remote, display).
                Default is full.

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


class ParameterGroup(QueryFuzzyList):
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
    def __init__(self, name, items, parent=None):
        self.name = name
        self.score = None
        self.parent = parent

        queryparams = []
        for item in items:
            this_param = self._make_query_parameter(item)
            queryparams.append(this_param)

        QueryFuzzyList.__init__(self, queryparams, use_fuzzy=get_best_fuzzy)
        self._check_names()
        if self.parent:
            self._check_datamodels()

    def __repr__(self):
        return '<ParameterGroup name={0.name}, n_parameters={1}>'.format(self, len(self))

    def __str__(self):
        return self.name

    def mapper(self, value):
        return value.full

    def _make_query_parameter(self, item):
        ''' Create and return a QueryParameter '''
        if isinstance(item, dict):
            item.update({'group': self.name, 'best': True})
            this_param = QueryParameter(**item)
        elif isinstance(item, six.string_types):
            is_best = [p for p in query_params.parameters
                       if item in p.full and (item == p.name or item == p.full)]
            if is_best:
                best_dict = is_best[0].__dict__
                best_dict.update({'group': self.name, 'best': True})
                this_param = QueryParameter(**best_dict)
            else:
                this_param = QueryParameter(item, group=self.name, best=False)
        if self.parent:
            this_param.set_parents(self.parent, self)

        return this_param

    def set_parent(self, parent):
        """Sets datamodel parent."""

        assert isinstance(parent, QueryDataModel), 'parent must be a QueryDataModel'
        self.parent = parent
        for item in self:
            item.set_parents(parent, self)

    @property
    def full(self):
        return self.list_params('full')

    @property
    def remote(self):
        return self.list_params('remote')

    @property
    def short(self):
        return self.list_params('short')

    @property
    def display(self):
        return self.list_params('display')

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

    def list_params(self, name_type=None, subset=None):
        ''' List the parameter names for a given group

        Lists the Query Parameters of the given group

        Parameters:
            subset (str|list):
                String list of a subset of parameters to return
            name_type (str):
                The type of name to generate (full, short, name, remote, display).

        Returns:
            param (list):
                The list of parameter
        '''

        assert name_type in ['full', 'short', 'name', 'remote', 'display', None], \
            'name_type must be (full, short, name, remote, display)'

        if subset:
            params = subset if isinstance(subset, list) else [subset]
            paramlist = [self[g] for g in params]
        else:
            paramlist = self

        if name_type:
            return [param.__getattribute__(name_type) for param in paramlist]
        else:
            return QueryList([param for param in paramlist])

    def _check_names(self):
        names = self.list_params('remote')
        for i, name in enumerate(names):
            if names.count(name) > 1:
                self[i].remote = self[i]._under
                self[i].name = self[i]._under
                self[i].short = self[i]._under
                self[i].display = '{0} {1}'.format(self[i].table.title(), self[i].display)

    def _check_datamodels(self):
        for item in self:
            if not item.property:
                if item.full in self.parent.dap_datamodel:
                    item.property = self.parent.dap_datamodel[item.full]


class QueryList(QueryFuzzyList):
    ''' A class for a list of Query Parameters '''

    def __init__(self, items):
        QueryFuzzyList.__init__(self, items, use_fuzzy=get_best_fuzzy)
        self._full = [s.full for s in self]
        self._remote = [s.remote for s in self]
        self._short = [s.short for s in self]

    def mapper(self, value):
        return value.full

    def get_full_from_remote(self, value):
        ''' Get the full name from the remote name '''
        return self._full[self._remote.index(value)]


class QueryParameter(object):
    ''' A Query Parameter class

    An object representing a query parameter.  Provides access to
    different names for a given parameter.

    Parameters:
        full (str):
            The full naming syntax (table.name) used for all queries.
            This name is recommended for full uniqueness.
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

    Attributes:
        property:
            The DAP Datamodel Property corresponding to this Query Parameter
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
        self.property = None
        self.parent = None
        self._set_names()

    def __repr__(self):
        return ('<QueryParameter full={0.full}, name={0.name}, short={0.short}, '
                'remote={0.remote}, display={0.display}>'.format(self))

    def __str__(self):
        return self.full

    def _split_full(self):
        ''' Split the full name into parts '''
        schema = table = column = None
        if self.full.count('.') == 1:
            table, column = self.full.split('.')
        elif self.full.count('.') == 2:
            schema, table, column = self.full.split('.')
        else:
            column = self.full

        return schema, table, column

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
            self.short = self._under if 'name' == self.name else self.name
        if not self.display:
            self.display = self.name.title()
        if not self.remote:
            self.remote = self._under if 'name' == self.name else self.name
        # correct the name if just name
        self.name = self._under if 'name' == self.name else self.name
        # used for a token string on the web
        self._joinedname = ', '.join([self.full, self.name, self.short, self.display])

    def set_parents(self, parent, group):
        ''' Set the parent datamodel '''

        assert isinstance(parent, QueryDataModel), 'parent must be a QueryDataModel'
        assert isinstance(group, ParameterGroup), 'group must be a ParameterGroup'
        self.parent = parent
        self.parent_group = group

    @property
    def db_schema(self):
        schema, table, column = self._split_full()
        if not schema:
            if self._in_form():
                schema = self.parent._marvinform._param_form_lookup[self.full].Meta.model.__table__.schema
            else:
                schema = None
        return schema

    @property
    def db_table(self):
        schema, table, column = self._split_full()
        release_num = self.parent.release.split('-')[1]
        table = table + release_num if table == 'spaxelprop' else table
        return table

    @property
    def db_column(self):
        schema, table, column = self._split_full()
        return column

    def is_hybrid(self):
        if self._in_form():
            model = self.parent._marvinform._param_form_lookup[self.full].Meta.model
            hybrids = get_hybrid_properties(model).keys()
            return self.db_column in hybrids
        return None

    def _in_form(self):
        ''' Check in parameters is in the Marvin Form '''
        if not hasattr(self.parent, '_marvinform'):
            return False

        return self.full in self.parent._marvinform._param_form_lookup


# # Get the Common Parameters from the filelist

def get_params():
    bestpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data',
                            'query_params_best.cfg')
    if os.path.isfile(bestpath):
        with open(bestpath, 'r') as stream:
            bestparams = yaml.load(stream, Loader=yamlordereddictloader.Loader)
        return bestparams
    else:
        return None


bestparams = get_params()
query_params = ParameterGroupList(bestparams)
