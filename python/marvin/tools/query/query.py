#!/usr/bin/env python
# encoding: utf-8

'''
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-02-17 14:13:28 by Brett Andrews
    2016-02-23 - Modified to test a programmatic query using a test sample form - B. Cherinka
    2016-03-02 - Generalized to many parameters and many forms - B. Cherinka
               - Added config drpver info
'''

from __future__ import print_function
from marvin.tools.core import MarvinToolsClass, MarvinError
from flask.ext.sqlalchemy import BaseQuery
from marvin import config, session, datadb
from marvin.tools.query.results import Results
from marvin.tools.query.forms import MarvinForm
from sqlalchemy import or_, and_, bindparam
from operator import le, ge, gt, lt, eq, ne
from collections import defaultdict
import re
from sqlalchemy.dialects import postgresql
from functools import wraps

__all__ = ['Query']
opdict = {'<=': le, '>=': ge, '>': gt, '<': lt, '!=': ne, '=': eq}
# opdict = {'le': le, 'ge': ge, 'gt': gt, 'lt': lt, 'ne': ne, 'eq': eq}


# decorator
def updateConfig(f):
    ''' Decorator that updates the query object with new config drpver version info '''

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if self.query:
            self.query = self.query.params({'drpver': config.drpver})
        return f(self, *args, **kwargs)
    return wrapper


class Query(object):
    ''' Core Marvin Query object.  can this be subclassed from sqlalchemy query? should it? '''

    def __init__(self, *args, **kwargs):

        # super(Query, self).__init__(*args, **kwargs) # potentially if we subclass query
        self.query = None
        self.params = None
        self.session = session
        self.filter = None
        self.joins = []
        self.myforms = defaultdict(str)

    def set_params(self, params=None):
        ''' Set parameters searched on into the query.  This updates a dictionary myforms with the appropriate form to
        modify/update based on the input parameters.  One-to-one mapping between parameter and form/modelclass/sqltable

        params = input dictonary of form {'parameter_name': parameter_value}
        e.g. params = {'nsa_redshift': u'< 0.012', 'name': u'19'}

        TODO: allow params to be a direct verbatim sql string, that should be parsed (into the above form?)

        '''

        if params:
            self.params = params

            print('query params', self.params)
            ''' params input should ideally be a multidict , then web/api+local versions can be same '''

            self.marvinform = MarvinForm()
            for key in self.params.keys():
                self.myforms[key] = self.marvinform.callInstance(self.marvinform._param_form_lookup[key], params=self.params)

    def add_condition(self):
        ''' Loop over all input forms and add a filter condition based on the input parameter form data. '''

        for form in self.myforms.values():
            # check if the SQL table already in the query, if not add it
            if not self._tableInQuery(form.Meta.model.__tablename__):
                self.joins.append(form.Meta.model.__tablename__)
                self.query = self.query.join(form.Meta.model)

            # build the actual filter
            self.build_filter(form)

        # add the filter to the query
        if not isinstance(self.filter, type(None)):
            self.query = self.query.filter(self.filter)

    def build_filter(self, form):
        ''' Builds a set of filter conditions to load into sqlalchemy filter.  Parameter data can take on form of

        'parameter_name': number            = assumes straight equality (e.g. 1.0 = redshift = 1.0)
        'parameter_name': operand number    = comparison operator and number (e.g. < 1.0 = redshift < 1.0)
        'parameter_name': number - number   = number range / between (e.g. 1-2 = redshifts >= 1 and <= 2)
        'parameter_name': number, number    = same as number range but comma separated

        '''

        for key, value in form.data.items():
            # Only do if a value is present
            if value:
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
                    myfilter = op(form.Meta.model.__table__.columns.__getitem__(key), number)
                else:
                    # do range or equality comparison
                    vals = re.split('[-,]', value.strip())
                    if len(vals) == 1:
                        # do straight equality comparison
                        number = vals[0]
                        if 'IFU' in str(form.Meta.model):
                            # Make the filter
                            myfilter = form.Meta.model.__table__.columns.__getitem__(key).startswith('{0}'.format(number))
                        else:
                            # Make the filter
                            myfilter = form.Meta.model.__table__.columns.__getitem__(key) == number
                    else:
                        # do range comparison
                        low, up = vals
                        # Make the filter
                        myfilter = and_(form.Meta.model.__table__.columns.__getitem__(key) >= low, form.Meta.model.__table__.columns.__getitem__(key) <= up)

                # Add new filter to the main filter
                if isinstance(self.filter, type(None)):
                    self.filter = and_(myfilter)
                else:
                    self.filter = and_(self.filter, myfilter)

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

        if qmode == 'all':
            res = self.query.all()
        elif qmode == 'one':
            res = self.query.one()
        elif qmode == 'first':
            res = self.query.first()
        elif qmode == 'count':
            res = self.query.count()

        return Results(results=res)

    @updateConfig
    def show(self, prop=None):
        ''' Prints info to the console with parameter variables plugged in.  '''

        assert prop in [None, 'query', 'tables', 'joins', 'filter'], 'Input must be query, joins, or filter'

        if not prop or 'query' in prop:
            print(self.query.statement.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))
        elif prop == 'tables':
            print(self.joins)
        elif prop == 'filter':
            print(self.filter.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))
        else:
            print(self.__getattribute__(prop))

    def reset(self):
        ''' Resets all query attributes '''

        self.filter = None
        self.myforms = None
        self.query = None
        self.params = None
        self.joins = None

    @updateConfig
    def _createBaseQuery(self, param=None):
        ''' Create the base query session object.  Default is to return a list of SQLalchemy Cube objects. Also default joins to the DRP pipeline
            version set from config.drpver
        '''

        if not param:
            self.query = self.session.query(datadb.Cube).join(datadb.PipelineInfo, datadb.PipelineVersion).filter(datadb.PipelineVersion.version == bindparam('drpver')).\
                params({'drpver': config.drpver})
        else:
            self.query = self.session.query(param)

    def _tableInQuery(self, name):
        ''' Checks if a given SQL table is already in the SQL query '''

        # check if a base query exists, this should probably be a decorator
        if not self.query:
            self._createBaseQuery()

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



