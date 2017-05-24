#!/usr/bin/env python
# encoding: utf-8

# Licensed under a 3-clause BSD license.

from __future__ import print_function, division
import re
import os
import numpy as np
import yaml
import yamlordereddictloader
from fuzzywuzzy import fuzz, process
from marvin import config
from marvin.core.exceptions import MarvinError

specindex_types = ['D4000', 'CaII0p39', 'HDeltaA', 'CN1', 'CN2', 'Ca4227', 'HGammaA', 'Fe4668',
                   'Hb', 'Mgb', 'Fe5270', 'Fe5335', 'Fe5406', 'NaD', 'TiO1', 'TiO2', 'NaI0p82',
                   'CaII0p86A', 'CaII0p86B', 'CaII0p86C', 'MgI0p88', 'TiO0p89', 'FeH0p99']

stkin_param_aliases = dict(VEL=['VEL', 'VELOCITY'],
                           SIGMA=['SIGMA', 'SIG'])

emline_name_aliases = dict(OIId3728=['OIId3728', 'OIId-3728', 'OII-3728', 'OII', 'OIId'],
                           Hb4862=['Hb4862', 'Hb-4862', 'Hb', 'Hbeta'],
                           OIII4960=['OIII4960', 'OIII-4960'],
                           OIII5008=['OIII5008', 'OIII-5008'],
                           NII6549=['NII6549', 'NII-6549'],
                           Ha6564=['Ha6564', 'Ha-6564', 'Ha', 'Halpha'],
                           NII6585=['NII6585', 'NII-6585'],
                           SII6718=['SII6718', 'SII-6718'],
                           SII6732=['SII6732', 'SII-6732'])
emline_param_aliases = dict(GFLUX=['GFLUX', 'FLUX'],
                            GVEL=['GVEL', 'VEL', 'VELOCITY'],
                            GSIMGA=['GSIGMA', 'SIG', 'SIGMA'],
                            EW=['EW', 'SEW'],
                            SFLUX=['SFLUX'],
                            INSTSIGMA=['INSTSIGMA'])

sp_lower = [it.lower() for it in specindex_types]
stkin_flat_lower = [i.lower() for v in stkin_param_aliases.values() for i in v]
emline_flat_lower = [i.lower() for v in emline_name_aliases.values() for i in v]
emline_param_flat_lower = [i.lower() for v in emline_param_aliases.values() for i in v]


def assign_category(name):
    category = None
    for frag in name.split('_'):
        nm_low = frag.lower()
        if nm_low in sp_lower:
            category = 'specindex'
            # Hb can be specindex or emline, so try finding an emline parameter in name.
            if nm_low == 'hb':
                for frag2 in name.split('_'):
                    if frag2.lower() in emline_param_flat_lower:
                        category = 'emline'
        elif nm_low == 'st':
            category = 'stellar_kin'
        elif nm_low in emline_flat_lower:
            category = 'emline'
    if category is None:
        raise TypeError
    return category


def assign_type(name, category):
    ctype = None
    for frag in name.split('_'):
        nm_low = frag.lower()
        if category == 'specindex':
            for sp, sp_low in zip(specindex_types, sp_lower):
                if nm_low == sp_low:
                    ctype = sp
        elif category == 'emline':
            for kk, vv in emline_name_aliases.items():
                for it in vv:
                    if nm_low == it.lower():
                        # split name--rest_wavelength string
                        ctype = list(re.findall(r'(\w+?)(\d+)', kk)[0])
                        ctype[1] = int(ctype[1])
    return ctype


def assign_parameter(name, category):
    param = None
    for frag in name.split('_'):
        nm_low = frag.lower()
        if category == 'stellar_kin':
            for kk, vv in stkin_param_aliases.items():
                for it in vv:
                    for frag2 in name.split('_'):
                        if frag2.lower() == it.lower():
                            param = kk
        elif category == 'emline':
            for kk, vv in emline_param_aliases.items():
                for it in vv:
                    if nm_low == it.lower():
                        param = kk
    return param


def assign_value_type(name):
    if 'ivar' in name:
        value_type = 'ivar'
    elif 'mask' in name:
        value_type = 'mask'
    else:
        value_type = 'value'
    return value_type


def join_type_conditions(ctype, category):
    if isinstance(ctype, str):
        ctype = [ctype]

    if ctype is None:
        conditions = None
    else:
        conditions = []
        for ct in ctype:
            if category == 'specindex':
                column = 'specindex_type.name'
            elif category == 'emline':
                if isinstance(ct, str):
                    column = 'emline_type.name'
                elif isinstance(ct, int):
                    column = 'emline_type.rest_wavelength'
            try:
                quote = "'" if isinstance(ct, str) else ''
                ctype_str = quote + str(ct) + quote
                conditions += [' '.join((column, '==', ctype_str))]
            except TypeError:
                conditions = None
    return conditions


def join_parameter_conditions(param, category):
    if category == 'stellar_kin':
        column = 'stellar_kin_parameter.name'
    elif category == 'emline':
        column = 'emline_parameter.name'
    try:
        param_str = "'" + param + "'"
        conditions = [' '.join((column, '==', param_str))]
    except TypeError:
        conditions = None
    return conditions


def expand(name, operator, value):
    category = assign_category(name)
    ctype = assign_type(name, category)
    param = assign_parameter(name, category)
    value_type = assign_value_type(name)
    value_table = '.'.join((category, value_type))
    # create searchfilter
    conditions = []
    type_condition = join_type_conditions(ctype, category)
    param_condition = join_parameter_conditions(param, category)
    value_condition = ' '.join((value_table, operator, value))
    for cn in (type_condition, param_condition, value_condition):
        if cn is not None:
            if isinstance(cn, list):
                for it in cn:
                    conditions.append(it)
            else:
                conditions.append(cn)
    searchfilter = ' and '.join(conditions)
    return searchfilter


# Query Parameter Datamodel

def get_best_fuzzy(name, choices, cutoff=0):
    items = process.extractBests(name, choices, score_cutoff=cutoff)
    if items:
        scores = [s[1] for s in items]
        morethanone = sum(np.max(scores) == scores) > 1
        if morethanone:
            exact = [s for s in items if s[0].name.lower() == name.lower()]
            if exact:
                return exact[0]
            else:
                options = [s[0].name for s in items if s[1] == np.max(scores)]
                raise KeyError('{0} is too ambiguous.  Did you mean one of {1}?'.format(name, options))
        else:
            return items[0]
    else:
        return None


def get_params():
    bestpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data', 'query_params_best_1.cfg')
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

    Methods:
        list_groups:
            lists all the groups in this list
        list_params:
            lists the (full) parameter names from all groups
            or a subset of groups
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

    def list_params(self, groups=None):
        '''Returns a list of parameters from all groups.

        Return a string list of the full parameter names.
        Default is all parameters across all groups.

        Parameters:
            groups (str|list):
                A string or list of strings representing the groups
                of parameters you wish to return

        Returns:
            params (list):
                A list of full parameter names
        '''
        if groups:
            groups = groups if isinstance(groups, list) else [groups]
            grouplist = [self[g] for g in groups]
            return [param.full for group in grouplist for param in group]
        else:
            return [param.full for group in self for param in group]

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

    Methods:
        list_params:
            List all the parameters for a given group

    '''
    def __init__(self, name, items):
        self.name = name
        self.score = None
        queryparams = [QueryParameter(**item) for item in items]
        list.__init__(self, queryparams)

    def __repr__(self):
        return ('<ParameterGroup name={0.name}, paramcount={1}>'.format(self, len(self)))

    def list_params(self, display=None, short=None, full=None):
        ''' List the parameter names

        Lists the Query Parameters of the given group

        Parameters:
            display (bool):
                Set to return the display names
            short (bool)
                Set to return the short names
            full (bool):
                Set to return the full names

        Returns:
            param (list):
                The list of parameter
        '''
        if short:
            return [param.short for param in self]
        elif display:
            return [param.display for param in self]
        elif full:
            return [param.full for param in self]
        else:
            return [param for param in self]

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
    '''

    def __init__(self, full, table=None, name=None, short=None, display=None):
        self.full = full
        self.table = table
        self.name = name
        self.short = short
        self.display = display
        self._joinedname = ', '.join([self.full, self.name, self.short, self.display])

    def __repr__(self):
        return ('<QueryParameter full={0.full}, name={0.name}, short={0.short}, display={0.display}>'.format(self))


bestparams = get_params()
query_params = ParameterGroupList(bestparams)



