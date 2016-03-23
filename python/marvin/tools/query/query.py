#!/usr/bin/env python
# encoding: utf-8

'''
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-02-17 14:13:28 by Brett Andrews
    2016-02-23 - Modified to test a programmatic query using a test sample form - B. Cherinka
    2016-03-02 - Generalized to many parameters and many forms - B. Cherinka
               - Added config drpver info
    2016-03-12 - Changed parameter input to be a natural language string
'''

from __future__ import print_function
from __future__ import division
from marvin.tools.core import MarvinToolsClass, MarvinError, MarvinUserWarning
from flask.ext.sqlalchemy import BaseQuery
from marvin import config, session, datadb
from marvin.tools.query.results import Results
from marvin.tools.query.forms import MarvinForm
from marvin.tools.query.modelGraph import ModelGraph
from sqlalchemy import or_, and_, bindparam, between
from operator import le, ge, gt, lt, eq, ne
from collections import defaultdict
import re
import warnings
from sqlalchemy.dialects import postgresql
from functools import wraps
from marvin.extern.sqlalchemy_boolean_search import (parse_boolean_search,
                                                     BooleanSearchException)

__all__ = ['Query']
opdict = {'<=': le, '>=': ge, '>': gt, '<': lt, '!=': ne, '=': eq}


# config.db = None
# session = None
# datadb = None


# Boom. Tree dictionary.
def tree():
    return defaultdict(tree)


# decorator
def updateConfig(f):
    ''' Decorator that updates the query object with new config drpver version info '''

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if self.query:
            self.query = self.query.params({'drpver': config.drpver})
        return f(self, *args, **kwargs)
    return wrapper


# decorator
def makeBaseQuery(f):
    ''' Decorator that makes the base query if it does not already exist '''

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.query:
            self._createBaseQuery()
        return f(self, *args, **kwargs)
    return wrapper


class Query(object):
    ''' Core Marvin Query object.  can this be subclassed from sqlalchemy query? should it? '''

    def __init__(self, *args, **kwargs):

        # super(Query, self).__init__(*args, **kwargs) # potentially if we subclass query
        self.query = None
        self.params = {}
        self.myparamtree = tree()
        self._paramtree = None
        self.session = session
        self.filter = None
        self.joins = []
        self.myforms = defaultdict(str)
        self.quiet = None
        self.mode = kwargs.get('mode', None)
        self._errors = []

        if self.mode is None:
            self.mode = config.mode

        if self.mode == 'local':
            self._doLocal()
        if self.mode == 'remote':
            self._doRemote()
        if self.mode == 'auto':
            self._doLocal()
            if self.mode == 'remote':
                self._doRemote()

    def _doLocal(self):
        ''' Tests if it is possible to perform queries locally. '''

        if not config.db or not self.session:
            warnings.warn('No local database found. Setting mode to remote', MarvinUserWarning)
            self.mode = 'remote'
        else:
            self.mode = 'local'

    def _doRemote(self):
        ''' Sets up to perform queries remotely. '''

        if not config.urlmap:
            raise MarvinError('No URL Map found.  Cannot make remote calls!')
        else:
            self.mode = 'remote'

    def set_filter(self, params=None):
        ''' Sets filter parameters searched on into the query.  This updates a dictionary myforms with the appropriate form to
        modify/update based on the input parameters.  One-to-one mapping between parameter and form/modelclass/sqltable

        Params is a string input of a boolean filter condition in SQL syntax
        e.g., params = " nsa_redshift < 0.012 and name = 19* "
        '''

        if params:
            # if params is a string, then parse and filter
            if type(params) == str or type(params) == unicode:
                try:
                    parsed = parse_boolean_search(params)
                except BooleanSearchException as e:
                    raise MarvinError('Your boolean expression contained a syntax error: {0}'.format(e))
            else:
                raise MarvinError('Input parameters must be a natural language string!')

            # update the parameters dictionary
            self._parsed = parsed
            self.strfilter = str(parsed)
            self.params.update(parsed.params)

            # print filter
            if not self.quiet:
                print('Your parsed filter is: ')
                print(parsed)

            # Perform local vs remote modes
            if self.mode == 'local':
                # Pass into Marvin Forms
                self._setForms()
            elif self.mode == 'remote':
                print('pass parameters to API here.  Need to figure out when and how to build a query remotely but still allow for user manipulation')

    def _setForms(self):
        ''' Set the appropriate WTForms in myforms and set the parameters '''
        self.marvinform = MarvinForm()
        self._paramtree = self.marvinform._paramtree
        for key in self.params.keys():
            self.myforms[key] = self.marvinform.callInstance(self.marvinform._param_form_lookup[key], params=self.params)
            self.myparamtree[self.myforms[key].Meta.model.__name__][key]

    def _validateForms(self):
        ''' Validate all the data in the forms '''

        formkeys = self.myforms.keys()
        isgood = [form.validate() for form in self.myforms.values()]
        if not all(isgood):
            inds = np.where(np.invert(isgood))[0]
            for index in inds:
                self._errors.append(self.myforms.values()[index].errors)
            raise MarvinError('Parameters failed to validate: {0}'.format(self._errors))

    def add_condition(self):
        ''' Loop over all input forms and add a filter condition based on the input parameter form data. '''

        self._validateForms()

        for form in self.myforms.values():
            # check if the SQL table already in the query, if not add it
            if not self._tableInQuery(form.Meta.model.__tablename__):
                self.joins.append(form.Meta.model.__tablename__)
                self.query = self.query.join(form.Meta.model)

        # build the actual filter
        self.build_filter()

        # add the filter to the query
        if not isinstance(self.filter, type(None)):
            self.query = self.query.filter(self.filter)

    def build_filter(self):
        ''' Builds a filter condition to load into sqlalchemy filter. '''
        self.filter = self._parsed.filter(datadb)

    """
    def build_filter(self, form):
        ''' Builds a set of filter conditions to load into sqlalchemy filter.  Parameter data can take on form of

        'parameter_name': number            = assumes straight equality (e.g. 1.0 = redshift = 1.0)
        'parameter_name': operand number    = comparison operator and number (e.g. < 1.0 = redshift < 1.0)
        'parameter_name': number - number   = number range / between (e.g. 1-2 = redshifts >= 1 and <= 2)
        'parameter_name': number, number    = same as number range but comma separated

        '''

        for key, value in form.data.items():
            # Only do if a value is present
            if value and not self._alreadyInFilter(key):
                # check for comparative operator
                iscompare = any([s in value for s in opdict.keys()])

                if iscompare:
                    # do operator comparison

                    # separate operator and value
                    value.strip()
                    try:
                        ops, number = value.split()
                    except ValueError:
                        match = re.match(r"([<>=!]+)([0-9.]+)", value, re.I)
                        if match:
                            ops = match.groups()[0]
                            number = match.groups()[1]
                    op = opdict[ops]

                    # Make the filter
                    myfilter = op(form.Meta.model.__table__.columns.__getitem__(key), bindparam(key, number))
                else:
                    # do range or equality comparison
                    vals = re.split('[-,]', value.strip()) if 'mangaid' not in key else [value.strip()]
                    if len(vals) == 1:
                        # do straight equality comparison
                        number = vals[0]
                        if 'IFU' in str(form.Meta.model):
                            # Make the filter
                            myfilter = form.Meta.model.__table__.columns.__getitem__(key).startswith(bindparam(key, number))
                        else:
                            # Make the filter
                            myfilter = form.Meta.model.__table__.columns.__getitem__(key) == bindparam(key, number)
                    else:
                        # do range comparison
                        low, up = vals
                        # Make the filter
                        myfilter = and_(between(form.Meta.model.__table__.columns.__getitem__(key), bindparam(key+'_1', low), bindparam(key+'_2', up)))

                # Add new filter to the main filter
                if isinstance(self.filter, type(None)):
                    self.filter = and_(myfilter)
                else:
                    self.filter = and_(self.filter, myfilter)
    """

    def update_params(self, param):
        ''' Update the input parameters '''
        param = {key: unicode(val) if '*' not in unicode(val) else unicode(val.replace('*', '%')) for key, val in param.items() if key in self.params.keys()}
        self.params.update(param)
        self._setForms()

    def _update_params(self, param):
        ''' this is now broken, this should update the boolean params in the filter condition '''

        ''' Update any input parameters that have been bound already.  Input is a dictionary of key, value pairs representing
            parameter name to update, and the value (number only) to update.  This does not allow to change the operand.
            Does not update self.params
            e.g.
            original input parameters {'nsa_redshift': '< 0.012'}
            newparams = {'nsa_redshift': '0.2'}
            update_params(newparams)
            new condition will be nsa_redshift < 0.2
        '''
        param = {key: unicode(val) if '*' not in unicode(val) else unicode(val.replace('*', '%')) for key, val in param.items() if key in self.params.keys()}
        self.query = self.query.params(param)

    def _alreadyInFilter(self, name):
        ''' Checks if the parameter name already added into the filter '''

        '''
        # my attempt at filtering on both parameter name and value; failed
        infilter = False
        if not isinstance(self.filter, type(None)):
            s = str(self.filter.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))
            splitfilter = s.split(name)
            if len(splitfilter) > 1:
                infilter = value in splitfilter[1]
        '''
        infilter = False
        if not isinstance(self.filter, type(None)):
            s = str(self.filter.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))
            infilter = name in s

        return infilter

    @makeBaseQuery
    @updateConfig
    def run(self, qmode='all'):
        ''' Run the query and return an instance of Marvin Results class to deal with results.  Input qmode allows to perform
            different sqlalchemy queries
        '''

        '''
        For remote mode, when and where does the API call occur?
            - Do they build the query here and send the query to the API?  If so, then they will need all sqlalchemy, and wtform, related
                package stuff installed.
            - Do they set only the parameters and send those to the API?
        '''

        # Check if params are set and there is a query
        if self.params and isinstance(self.query.whereclause, type(None)):
            self.add_condition()

        # get total count, and if more than 150 results, paginate and only return the first 10
        count = self.query.count()
        if count > 150:
            self.query = self.query.slice(0, 10)
            warnings.warn('Results contain more than 150 entries.  Only returning first 10', MarvinUserWarning)

        if qmode == 'all':
            res = self.query.all()
        elif qmode == 'one':
            res = self.query.one()
        elif qmode == 'first':
            res = self.query.first()
        elif qmode == 'count':
            res = self.query.count()

        return Results(results=res, query=self.query, count=count)

    @updateConfig
    def show(self, prop=None):
        ''' Prints info to the console with parameter variables plugged in.  '''

        assert prop in [None, 'query', 'tables', 'joins', 'filter'], 'Input must be query, joins, or filter'

        if self.mode == 'local':
            if not prop or 'query' in prop:
                print(self.query.statement.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))
            elif prop == 'tables':
                print(self.joins)
            elif prop == 'filter':
                '''oddly this does not update when bound parameters change, but the statement above does '''
                print(self.query.whereclause.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))
            else:
                print(self.__getattribute__(prop))
        elif self.mode == 'remote':
            print('Cannot show full SQL query in remote mode, use the API')

    def reset(self):
        ''' Resets all query attributes '''
        self.__init__()

    @updateConfig
    def _createBaseQuery(self, param=None):
        ''' Create the base query session object.  Default is to return a list of SQLalchemy Cube objects. Also default joins to the DRP pipeline
            version set from config.drpver
        '''

        if not param:
            self.query = self.session.query(datadb.Cube).join(datadb.PipelineInfo, datadb.PipelineVersion).filter(datadb.PipelineVersion.version == bindparam('drpver', config.drpver))
        else:
            self.query = self.session.query(param).join(datadb.PipelineInfo, datadb.PipelineVersion).filter(datadb.PipelineVersion.version == bindparam('drpver', config.drpver))

    @makeBaseQuery
    def _tableInQuery(self, name):
        ''' Checks if a given SQL table is already in the SQL query '''

        # do the check
        try:
            isin = name in str(self.query._from_obj[0])
        except IndexError as e:
            isin = False
        except AttributeError as e:
            if type(self.query) == str:
                isin = name in self.query
            else:
                isin = False
        return isin
