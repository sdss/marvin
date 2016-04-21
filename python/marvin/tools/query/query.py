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
from marvin.core import MarvinToolsClass, MarvinError, MarvinUserWarning
from sqlalchemy_boolean_search import parse_boolean_search, BooleanSearchException
from marvin import config, marvindb
from marvin.tools.query.results import Results
from marvin.tools.query.forms import MarvinForm
from brain.api.api import Interaction
from sqlalchemy import or_, and_, bindparam, between
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.expression import desc
from sqlalchemy.ext.declarative import DeclarativeMeta
from operator import le, ge, gt, lt, eq, ne
from collections import defaultdict
import re
import warnings
from functools import wraps

__all__ = ['Query', 'doQuery']
opdict = {'<=': le, '>=': ge, '>': gt, '<': lt, '!=': ne, '=': eq}


# Boom. Tree dictionary.
def tree():
    return defaultdict(tree)


# Do A Query
def doQuery(searchfilter, limit=10, sort=None, order=None):
    q = Query(limit=limit, sort=sort, order=order)
    q.set_filter(searchfilter=searchfilter)
    q.add_condition()
    res = q.run()
    return q, res


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


# decorator
def checkCondition(f):
    ''' Decorator that checks to ensure the filter is set in the property, if it does not already exist '''

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if self.filterparams and not self._alreadyInFilter(self.filterparams.keys()):
            self.add_condition()
        return f(self, *args, **kwargs)

    return wrapper


class Query(object):
    ''' Core Marvin Query object.  can this be subclassed from sqlalchemy query? should it? '''

    def __init__(self, *args, **kwargs):

        self.query = None
        self.params = []
        self.filterparams = {}
        self.myparamtree = tree()
        self._paramtree = None
        self.session = marvindb.session
        self.filter = None
        self.joins = []
        self.myforms = defaultdict(str)
        self.quiet = None
        self._errors = []
        self._basetable = None
        self._modelgraph = marvindb.modelgraph
        self.mode = kwargs.get('mode', None)
        self.limit = int(kwargs.get('limit', 10))
        self.sort = kwargs.get('sort', None)
        self.order = kwargs.get('order', 'asc')
        self.marvinform = MarvinForm()

        # set the mode
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

        # get return type
        self.returntype = kwargs.get('returntype', None)

        # get user-defined input parameters
        inputparams = kwargs.get('inputparams', None)
        if inputparams:
            self.set_inputparams(inputparams)

        # if searchfilter is set then set the parameters
        searchfilter = kwargs.get('searchfilter', None)
        if searchfilter:
            self.set_filter(searchfilter=searchfilter)

        # create query parameter ModelClasses
        self._create_query_modelclasses()

        # join tables
        self._join_tables()

        # add condition
        if searchfilter:
            self.add_condition()

    def __repr__(self):
        return ('Query(mode={0}, limit={1}, sort={2}, order={3})'
                .format(repr(self.mode), self.limit, self.sort, repr(self.order)))

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

    def set_inputparams(self, inputparams):
        ''' Loads the user input parameters into the query params limit '''
        self.params.extend(inputparams)
        self.params = list(set(self.params))

    def set_defaultparams(self):
        ''' Loads the default params for a given return type '''
        pass

    def _create_query_modelclasses(self):
        ''' Creates a list of database ModelClasses from a list of parameter names '''
        self.queryparams = self.marvinform._param_form_lookup.mapToColumn(self.params)

    def set_filter(self, searchfilter=None):
        ''' Sets filter parameters searched on into the query.  This updates a dictionary myforms
        with the appropriate form to modify/update based on the input parameters.  One-to-one
        mapping between parameter and form/modelclass/sqltable

        Params is a string input of a boolean filter condition in SQL syntax
        e.g., params = " nsa_redshift < 0.012 and name = 19* "
        '''

        if searchfilter:
            # if params is a string, then parse and filter
            if type(searchfilter) == str or type(searchfilter) == unicode:
                try:
                    parsed = parse_boolean_search(searchfilter)
                except BooleanSearchException as e:
                    raise MarvinError('Your boolean expression contained a syntax error: {0}'.format(e))
            else:
                raise MarvinError('Input parameters must be a natural language string!')

            # update the parameters dictionary
            self.searchfilter = searchfilter
            self._parsed = parsed
            self.strfilter = str(parsed)
            self.filterparams.update(parsed.params)
            self.params.extend(self.filterparams.keys())

            # print filter
            if not self.quiet:
                print('Your parsed filter is: ')
                print(parsed)

            # Perform local vs remote modes
            if self.mode == 'local':
                # Pass into Marvin Forms
                try:
                    self._setForms()
                except KeyError as e:
                    self.reset()
                    raise MarvinError('Could not set parameters. Multiple entries found for key.  Be more specific: {0}'.format(e))
            elif self.mode == 'remote':
                # Is it possible to build a query remotely but still allow for user manipulation?
                pass

    def _setForms(self):
        ''' Set the appropriate WTForms in myforms and set the parameters '''
        self._paramtree = self.marvinform._paramtree
        for key in self.filterparams.keys():
            self.myforms[key] = self.marvinform.callInstance(self.marvinform._param_form_lookup[key], params=self.filterparams)
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

        # validate the forms
        self._validateForms()

        # build the actual filter
        self.build_filter()

        # add the filter to the query
        if not isinstance(self.filter, type(None)):
            self.query = self.query.filter(self.filter)

    @makeBaseQuery
    def _join_tables(self):
        ''' Build the join statement from the input parameters '''
        mymodellist = [param.class_ for param in self.queryparams]

        # Gets the list of joins from ModelGraph. Uses Cube as nexus, so that
        # the order of the joins is the correct one.
        # TODO: at some point, all the queries should be generalised so that
        # we don't assume that we are querying a cube.
        self._modellist = self._modelgraph.getJoins(mymodellist, format_out='models', nexus=marvindb.datadb.Cube)

        # sublist = [model for model in modellist if model.__tablename__ not in self._basetable and not self._tableInQuery(model.__tablename__)]
        # self.joins.extend([model.__tablename__ for model in sublist])
        # self.query = self.query.join(*sublist)
        for model in self._modellist:
            if not self._tableInQuery(model.__tablename__):
                self.joins.append(model.__tablename__)
                self.query = self.query.join(model)

    def build_filter(self):
        ''' Builds a filter condition to load into sqlalchemy filter. '''
        self.filter = self._parsed.filter(self._modellist)

    def update_params(self, param):
        ''' Update the input parameters '''
        param = {key: unicode(val) if '*' not in unicode(val) else unicode(val.replace('*', '%')) for key, val in param.items() if key in self.filterparams.keys()}
        self.filterparams.update(param)
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
        param = {key: unicode(val) if '*' not in unicode(val) else unicode(val.replace('*', '%')) for key, val in param.items() if key in self.filterparams.keys()}
        self.query = self.query.params(param)

    def _alreadyInFilter(self, names):
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
        '''
        infilter = False
        if not isinstance(self.filter, type(None)):
            s = str(self.filter.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))
            infilter = name in s
        '''

        infilter = None
        if names:
            if not isinstance(self.query, type(None)):
                if not isinstance(self.query.whereclause, type(None)):
                    wc = str(self.query.whereclause)
                    infilter = any([name in wc for name in names])

        return infilter

    @makeBaseQuery
    @checkCondition
    @updateConfig
    def run(self, qmode='all'):
        ''' Run the query and return an instance of Marvin Results class to deal with results.  Input qmode allows to perform
            different sqlalchemy queries
        '''

        if self.mode == 'local':

            # Check if filter params are set and there is a query
            # if self.filterparams and isinstance(self.query.whereclause, type(None)):
            #     print('adding conditions')
            #     self.add_condition()

            # Check for adding a sort
            self._sortQuery()

            # get total count, and if more than 150 results, paginate and only return the first 10
            count = self.query.count()
            if count > 150:
                query = self.query.slice(0, self.limit)
                warnings.warn('Results contain more than 150 entries.  Only returning first 10', MarvinUserWarning)
            else:
                query = self.query

            if qmode == 'all':
                res = query.all()
            elif qmode == 'one':
                res = query.one()
            elif qmode == 'first':
                res = query.first()
            elif qmode == 'count':
                res = query.count()

            return Results(results=res, query=self.query, count=count, mode=self.mode)

        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['querycubes']['url']

            params = {'searchfilter': self.searchfilter}
            try:
                ii = Interaction(route=url, params=params)
            except MarvinError as e:
                raise MarvinError('API Query call failed: {0}'.format(e))
            else:
                res = ii.results
            return Results(results=res, query=self.query, mode=self.mode)

    def _sortQuery(self):
        ''' Sort the query by a given parameter '''
        if not isinstance(self.sort, type(None)):
            # set the sort variable ModelClass parameter ; make this as plateifu right now
            # TODO - generalize this to any parameter
            if self.sort == 'plateifu':
                sortparam = marvindb.datadb.Cube.plateifu
                if not self._tableInQuery('ifudesign'):
                    self.query = self.query.join(marvindb.datadb.IFUDesign)

            # If order is specified, then do the sort
            if self.order:
                assert self.order in ['asc', 'desc'], 'Sort order parameter must be either "asc" or "desc"'

                # Check if order by already applied
                if 'ORDER' in str(self.query.statement):
                    self.query = self.query.order_by(None)
                # Do the sorting
                if 'desc' in self.order:
                    self.query = self.query.order_by(desc(sortparam))
                else:
                    self.query = self.query.order_by(sortparam)

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
    def _createBaseQuery(self):
        ''' Create the base query session object.  Default is to return a list of SQLalchemy Cube objects. Also default joins to the DRP pipeline
            version set from config.drpver
        '''

        # if not param:
        #     param = marvindb.datadb.Cube
        #     self._basetable = self._buildBaseTable(param)
        #     self.query = self.session.query(marvindb.datadb.Cube).join(marvindb.datadb.PipelineInfo, marvindb.datadb.PipelineVersion)\
        #         .filter(marvindb.datadb.PipelineVersion.version == bindparam('drpver', config.drpver))
        # else:
        #     self._basetable = self._buildBaseTable(param)
        #     self.query = self.session.query(param).join(marvindb.datadb.PipelineInfo, marvindb.datadb.PipelineVersion)\
        #         .filter(marvindb.datadb.PipelineVersion.version == bindparam('drpver', config.drpver))
        self.query = self.session.query(*self.queryparams).join(marvindb.datadb.PipelineInfo, marvindb.datadb.PipelineVersion)\
            .filter(marvindb.datadb.PipelineVersion.version == bindparam('drpver', config.drpver))

    def _buildBaseTable(self, param):
        ''' Builds the base name for a input parameter: either schema.table or schema.table.column'''
        if isinstance(param, DeclarativeMeta) and hasattr(param, '__table__'):
            basename = '{0}.{1}'.format(param.__table__.schema, param.__table__.name)
        elif isinstance(param, InstrumentedAttribute):
            basename = str(param.compile())
            basename = basename.rsplit('.', 1)[0]
        else:
            basename = None
        return basename

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
