#!/usr/bin/env python
# encoding: utf-8

'''
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-02-23 14:13:28 by Brian Cherinka
    2016-02-23 - made some test sample forms with wtforms+sqlalchemy - B. Cherinka
    2016-03-02 - generalized the form to build all forms - B. Cherinka
'''

from __future__ import division, print_function

import re
import sys
import warnings
from collections import OrderedDict, defaultdict
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.inspection import inspect as sa_inspect
from sqlalchemy.orm.attributes import InstrumentedAttribute
from wtforms import Field, SelectMultipleField, StringField, SubmitField, ValidationError, validators
from wtforms.widgets import TextInput
from wtforms_alchemy import model_form_factory
from marvin import config, marvindb
from marvin.core.exceptions import MarvinUserWarning
from marvin.utils.general.structs import FuzzyDict

# flask_wtf does not work locally - OUTSIDE APPLICATON CONTEXT;
# need some kind of toggle for web version and not???
if config._inapp:
    from flask_wtf import Form
else:
    from wtforms import Form

__all__ = ['MarvinForm']


def tree():
    return defaultdict(tree)


# Base form class
BaseModelForm = model_form_factory(Form)


class ModelForm(BaseModelForm):
    ''' sub class a new WTF ModelForm so it works with Flask-WTF in APP mode;
    for auto CSRF tokens...who knows...
    '''
    @classmethod
    def get_session(self):
        return marvindb.session

# Builds a dictionary for modelclasses with key ClassName and value SQLalchemy model class
# modelclasses = marvindb.buildUberClassDict()


# Class factory
def formClassFactory(name, model, baseclass):
    ''' Generates a new WTForm Class based on SQLalchemy Model Class.

    Subclasses a base WTF Form class that also contains the SQLAlchemy
    Model Class information inside it.

    Each class contains as attributes:
        Meta = a class called Meta.  Meta.model contains the SQLalchemy ModelClass
        data = a dictionary of parameters: form input that gets mapped to the sqlalchemy parameter
        errors = a dictionary of errors returned by invalid form validation
        validate = a method to validate all elements in this form
        parameter_X = a WTForm Field mapped to respective sqlalchemy table column

        e.g.
        The ModelClass IFUDesign mapped to mangadatadb.ifu_design sql table gets transformed into
        WTForm IFUDesignForm, with IFUDesignForm.Meta.model = marvin.db.models.DataModelClasses.IFUDesign

    Parameters:
        name (str):
            The name of the Form Class
        mdoel (class):
            The SQLAlchemy Model Class
        baseclass (class):
            The base class to sub class from

    Returns:
        the new WTF form subclass
    '''

    Meta = type('Meta', (object,), {'model': model})
    newclass = type(name, (baseclass,), {'Meta': Meta})
    return newclass

# build a wtform select field for operators; tested but no longer used
# can't seem to attach operator field to every individual parameter
opdict = {'le': '<=', 'ge': '>=', 'gt': '>', 'lt': '<', 'ne': '!=', 'eq': '='}
ops = [(key, val) for key, val in opdict.items()]
# operator = SelectField(u'Operator', choices=ops)


class ParamFxnLookupDict(dict):
    ''' Parameter function lookup for new function expressions
    '''

    def __getitem__(self, key):

        lowkey = key.lower()
        mykeys = list(self.keys())

        inkey = lowkey in mykeys
        if not inkey:
            raise KeyError('{0} does not match any column.'.format(lowkey))
        else:
            keycount = mykeys.count(lowkey)
            if keycount > 1:
                raise KeyError('{0} matches multiple parameters in the lookup'
                               ' table'.format(lowkey))
            else:
                return dict.__getitem__(self, lowkey)


class ParamFormLookupDict(dict):

    def __init__(self, **kwargs):
        self.allspaxels = kwargs.get('allspaxels', None)
        self._release = kwargs.get('release', config.release)
        self._init_shortcuts()

    def __contains__(self, value):
        ''' Override the contains '''

        try:
            key = self[value]
        except KeyError as e:
            key = None

        return key is not None

    def __getitem__(self, key):
        """Checks if `key` is a unique column name and return the value."""

        # Init the shortcuts
        self._init_shortcuts()

        # Applies shortcuts
        keySplits = self._apply_shortcuts(key)

        # Get key matches
        matches = self._get_matches(keySplits)

        # Check DAP Junk keys
        matches = self._check_for_junk(matches)

        # Return matched key
        return dict.__getitem__(self, self._get_good_match(key, matches))

    def _check_for_junk(self, matches):
        ''' check for Junk matches and return the correct key '''
        isdapjunk = any(['mangadapdb.spaxelprop' in m for m in matches])
        if isdapjunk:
            junkmatches = [m for m in matches if (self.allspaxels and 'clean'
                           not in m) or (not self.allspaxels and 'clean' in m)]
            keySplits = self._apply_shortcuts(junkmatches[0])
            matches = self._get_matches(keySplits)
        return matches

    def _get_good_match(self, key, matches):
        ''' Check if the key match is good '''
        if len(matches) == 0:
            # No matches. This returns the standards KeyError from dict
            raise KeyError('{0} does not match any column.'.format(key))
        elif len(matches) == 1:
            # One match: returns True
            return matches[0]
        else:
            # Multiple results. Raises a custom error.
            raise KeyError(
                '{0} matches multiple parameters in the lookup table: {1}'
                .format(key, ', '.join(matches)))

    def mapToColumn(self, keys):
        """Returns the model class column in the WFTForm."""

        if not isinstance(keys, (list, tuple)):
            str_types = [str]
            if sys.version_info[0] < 3:
                str_types.append(unicode)
            if isinstance(keys, tuple(str_types)):
                keys = [keys]
            else:
                keys = list(keys)

        columns = []
        for key in keys:
            keySplits = self._apply_shortcuts(key)
            matches = self._get_matches(keySplits)
            matches = self._check_for_junk(matches)
            key = self._get_good_match(key, matches)
            wtfForm = self[key]
            column = key.split('.')[-1]
            columns.append(getattr(wtfForm.Meta.model, column))

        if len(columns) == 1:
            return columns[0]
        else:
            return columns

    def _init_table_shortcuts(self):
        ''' initialize the table shortcuts '''

        self._schemaShortcuts = {'datadb': 'mangadatadb', 'dapdb': 'mangadapdb', 'sampledb': 'mangasampledb',
                                 'auxdb': 'mangaauxdb'}

        self._tableShortcuts = {'ifu': 'ifudesign', 'cube_header_keyword': 'fits_header_keyword',
                                'cube_header_value': 'fits_header_value', 'maps_header_keyword': 'header_keyword',
                                'maps_header_value': 'header_value'}
        self._set_junk_shortcuts()

    def _init_name_shortcuts(self):
        ''' initialize the name shortcuts

        e.g {'haflux': 'emline_gflux_ha_6564',
             'g_r': 'elpetro_mag_g_r',
             'abs_g_r': 'elpetro_absmag_g_r'}

        '''
        from marvin.utils.datamodel.query.base import query_params
        short = query_params.list_params(name_type='short')
        name = query_params.list_params(name_type='name')
        self._nameShortcuts = OrderedDict(zip(short, name))

    def _init_full_shortcuts(self):
        ''' initialize any full name shortcuts '''
        self._fullShortcuts = {'cube_header.keyword':'fits_header_keyword.label',
                               'cube_header.value':'fits_header_value.value',
                               'maps_header.keyword':'header_keyword.name',
                               'maps_header.value':'header_value.value'}

    def _init_shortcuts(self):
        ''' initialize the shortcuts '''
        self._init_full_shortcuts()
        self._init_table_shortcuts()
        self._init_name_shortcuts()

    def _apply_shortcuts(self, key):
        ''' Apply the shortcuts to the key '''

        # Apply any full name shortcuts
        if key in self._fullShortcuts:
            return self._fullShortcuts[key].split('.')

        # Gets the paths that match the key
        keySplits = key.split('.')

        # Apply schema shortcuts
        if len(keySplits) >= 3 and keySplits[-3] in self._schemaShortcuts:
            keySplits[-3] = self._schemaShortcuts[keySplits[-3]]

        # Applies table shortcuts
        if len(keySplits) >= 2 and keySplits[-2] in self._tableShortcuts:
            keySplits[-2] = self._tableShortcuts[keySplits[-2]]

        # Applies name shortcuts
        if len(keySplits) >= 1 and keySplits[-1] in self._nameShortcuts:
            keySplits[-1] = self._nameShortcuts[keySplits[-1]]

        return keySplits

    def get_real_name(self, key):
        ''' Returns the real full key given some shortcut names '''
        keySplits = self._apply_shortcuts(key)
        return '.'.join(keySplits)

    def get_shortcut_name(self, key):
        ''' Returns the shortcutted full name given a real key '''
        pass

    def _set_junk_shortcuts(self):
        ''' Sets the DAP spaxelprop shortcuts based on MPL '''

        spaxname = 'spaxelprop' if self.allspaxels else 'cleanspaxelprop'

        # available spaxel property tables
        dapcut = {}
        spdict = marvindb.spaxelpropdict
        if spdict:
            # create a shortcut for each table that points to the current release
            current = spdict.get(self._release, None)
            for release, name in spdict.items():
                if current:
                    cut = {name.lower(): current.lower().replace('spaxelprop', spaxname)}
                    dapcut.update(cut)

            # since for DRs spaxelprop may not be in the dictionary a la MPL4: SpaxelProp
            # we need to add it in so the shortcut "spaxelprop" can correctly point to the right db param
            if 'spaxelprop' not in dapcut and current:
                dapcut.update({'spaxelprop': current.lower().replace('spaxelprop', spaxname)})

        # update the main dictionary
        self._tableShortcuts.update(dapcut)

    def _get_matches(self, keySplits):
        ''' Get the matches from a set of key splits '''
        matches = [path for path in self
                   if all([keySplits[-1 - ii] == path.split('.')[-1 - ii]
                           for ii in range(len(keySplits))])]
        return matches


# Custom [Better]TagList Field for Search Parameter Box
class TagListField(Field):
    widget = TextInput()

    def _value(self):
        if self.data:
            return u', '.join(self.data)
        else:
            return u''

    def process_formdata(self, valuelist):
        if valuelist:
            self.data = [x.strip() for x in valuelist[0].split(',') if x]
        else:
            self.data = []


class BetterTagListField(TagListField):
    def __init__(self, label='', validators=None, remove_duplicates=True, **kwargs):
        super(BetterTagListField, self).__init__(label, validators, **kwargs)
        self.remove_duplicates = remove_duplicates

    def process_formdata(self, valuelist):
        super(BetterTagListField, self).process_formdata(valuelist)
        if self.remove_duplicates:
            self.data = list(self._remove_duplicates(self.data))

    @classmethod
    def _remove_duplicates(cls, seq):
        """Remove duplicates in a case insensitive, but case preserving manner"""
        dups = {}
        for item in seq:
            if item.lower() not in dups:
                dups[item.lower()] = True
                yield item


# Custom validator for MainForm
class ValidOperand(object):
    def __init__(self, opstring='[<>=]', message=None):
        self.opstring = opstring
        if not message:
            message = u'Field must contain at least a valid operand of {0}.'.format(self.opstring)
        self.message = message

    def __call__(self, form, field):
        infield = re.search(self.opstring, field.data)
        if not infield:
            raise ValidationError(self.message)


class SearchForm(Form):
    ''' Main Search Level WTForm for Marvin '''

    searchbox = StringField("<a target='_blank' href='https://sdss-marvin.readthedocs.io/en/stable/tools/query/query_using.html'>Input Search Filter</a>",
        [validators.Length(min=3, message='Input must have at least 3 characters'),
                            validators.DataRequired(message='Input filter string required'),
                            ValidOperand('[<>=between]', message='Input must contain a valid operand.')])
    returnparams = SelectMultipleField("<a target='_blank' href='https://api.sdss.org/doc/manga/marvin/query_params.html'>Return Parameters</a>")
    submitsearch = SubmitField('Search')


class MarvinForm(object):
    ''' Core Marvin Form object. '''

    def __init__(self, *args, **kwargs):
        ''' Initializes a Marvin Form

        Generates all the WTForms from the SQLAlchemy ModelClasses defined in the MaNGA DB.

        _param_form_lookup = dictionary of all modelclass parameters
        of form {'SQLalchemy ModelClass parameter name': WTForm Class}
        '''

        self._release = kwargs.get('release', config.release)
        self.verbose = kwargs.get('verbose', False)
        if marvindb:
            self._modelclasses = FuzzyDict(marvindb.buildUberClassDict(release=self._release))
            self._param_form_lookup = ParamFormLookupDict(**kwargs)
            self._param_fxn_lookup = ParamFxnLookupDict()
            self._paramtree = tree()
            self._generateFormClasses(self._modelclasses)
            self._generateFxns()
            self.SearchForm = SearchForm
            self._cleanParams(**kwargs)

    def __repr__(self):
        nforms = len([f for f in self.__dict__.keys() if 'Form' in f])
        return ('<MarvinForm (release={0._release}, n_parameters={1}, n_functions={2}, '
                'n_forms={3})>'.format(self, len(self._param_form_lookup), len(self._param_fxn_lookup), nforms))

    def _generateFormClasses(self, classes):
        ''' Loops over all ModelClasses and generates a new WTForm class.  New form classes are named as [ModelClassName]Form.
            Sets the new form as an attribute on MarvinForm.  Also populates the _param_to_form_lookup dictonary with
            all ModelClass/WTForm parameters and their corresponding forms.

            e.g.  _param_form_lookup['name'] = marvin.tools.query.forms.IFUDesignForm
        '''

        for key, val in classes.items():
            classname = '{0}Form'.format(key)
            try:
                newclass = formClassFactory(classname, val, ModelForm)
            except Exception as e:
                if self.verbose:
                    warnings.warn('class {0} not Formable'.format(key), MarvinUserWarning)
            else:
                self.__setattr__(classname, newclass)
                self._loadParams(newclass)

    def _loadParams(self, newclass):
        ''' Loads all parameters from wtforms into a dictionary with
            key, value = {'parameter_name': 'parent WTForm name'}.
            Ignores hidden attributes and the Meta class
        '''

        model = newclass.Meta.model
        schema = model.__table__.schema
        tablename = model.__table__.name

        mapper = sa_inspect(model)
        for key, item in mapper.all_orm_descriptors.items():
            if isinstance(item, (hybrid_property, hybrid_method)):
                key = key
            elif isinstance(item, InstrumentedAttribute):
                key = item.key
            else:
                continue

            lookupKeyName = schema + '.' + tablename + '.' + key
            self._param_form_lookup[lookupKeyName] = newclass
            self._paramtree[newclass.Meta.model.__name__][key]

    def callInstance(self, form, params=None, **kwargs):
        ''' Creates an instance of a specified WTForm.  '''
        return form(**params) if params else form(**kwargs)

    def _generateFxns(self):
        ''' Generate the fxn dictionary

            TODO: make this general and not hard-coded

            The key is the function name used in the query syntax.
            The value is the method call that lives in Query

        '''
        self._param_fxn_lookup['npergood'] = '_get_percent'
        self._param_fxn_lookup['radial'] = '_radial_query'

    def _getDapKeys(self):
        ''' Returns the DAP keys from the Junk tables
        '''
        dapkeys = [k for k in self._param_form_lookup.keys() if 'mangadapdb.spaxelprop' in k]
        dapkeys.sort()
        return dapkeys

    def _cleanParams(self, **kwargs):
        ''' Clean up the parameter-form lookup dictionary '''

        # remove keys for pk, mangadatadb.sample, test_, and cube_header
        new = ParamFormLookupDict(**kwargs)
        for k, v in self._param_form_lookup.items():
            if 'pk' not in k and \
               'mangadatadb.sample' not in k and \
               'test_' not in k and \
               'cube_header' not in k:
                new[k] = v

        # make new dictionary
        self._param_form_lookup = new

    def look_up_table(self, table):
        ''' Look up a database table and return the ModelClass '''

        try:
            table_class = self._modelclasses[table]
        except (ValueError, TypeError) as e:
            table_class = None

        return table_class
