#!/usr/bin/env python
# encoding: utf-8

# Licensed under a 3-clause BSD license.

# Revision History:
#     Initial Version: 2016-02-17 14:13:28 by Brett Andrews
#     2016-02-23 - Modified to test a programmatic query using a test sample form - B. Cherinka
#     2016-03-02 - Generalized to many parameters and many forms - B. Cherinka
#                - Added config drpver info
#     2016-03-12 - Changed parameter input to be a natural language string

from __future__ import print_function
from __future__ import division
from marvin.core import MarvinToolsClass, MarvinError, MarvinUserWarning
from sqlalchemy_boolean_search import parse_boolean_search, BooleanSearchException
from marvin import config, marvindb
from marvin.tools.query.results import Results
from marvin.tools.query.forms import MarvinForm
from marvin.api.api import Interaction
from sqlalchemy import or_, and_, bindparam, between
from sqlalchemy.orm import aliased
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
def doQuery(*args, **kwargs):
    ''' Convenience function for Query+Results

        A wrapper for performing a Query, running it, and retrieving
        the Results.

        Parameters:
            see the Query class for a list of inputs

        Returns:
            query, results:
                A tuple containing the built Query instance, and the results
                instance
    '''
    q = Query(*args, **kwargs)
    res = q.run()
    return q, res


# decorator
def updateConfig(f):
    ''' Decorator that updates the query object with new config drpver version info '''

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if self.query and self.mode == 'local':
            self.query = self.query.params({'drpver': config.drpver, 'dapver': config.dapver})
        return f(self, *args, **kwargs)
    return wrapper


# decorator
def makeBaseQuery(f):
    ''' Decorator that makes the base query if it does not already exist '''

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.query and self.mode == 'local':
            self._createBaseQuery()
        return f(self, *args, **kwargs)
    return wrapper


# decorator
def checkCondition(f):
    ''' Decorator that checks to ensure the filter is set in the property, if it does not already exist '''

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if self.mode == 'local' and self.filterparams and not self._alreadyInFilter(self.filterparams.keys()):
            self.add_condition()
        return f(self, *args, **kwargs)

    return wrapper


class Query(object):
    ''' A class to perform queries on the MaNGA dataset.

    This class is the main way of performing a query.  A query works minimally
    by specifying a list of desired parameters, along with a string filter
    condition in a natural language SQL format.

    A local mode query assumes a local database.  A remote mode query uses the
    API to run a query on the Utah server, and return the results.

    By default, the query returns a list of tupled parameters.  The parameters
    are a combination of user-defined parameters, parameters used in the
    filter condition, and a set of pre-defined default parameters.  The object
    plate-IFU or mangaid is always returned by default.

    Parameters:
        returnparams (str list):
            A list of string parameters names desired to be returned in the query
        searchfilter (str):
            A (natural language) string containing the filter conditions
            in the query; written as you would say it.
        returntype (str):
            The requested Marvin Tool object that the results are converted into
        mode ({'local', 'remote', 'auto'}):
            The load mode to use. See :doc:`Mode secision tree</mode_decision>`.
        sort (str):
            The parameter name to sort the query on
        order ({'asc', 'desc'}):
            The sort order.  Can be either ascending or descending.
        limit (int):
            The number limit on the number of returned results

    Returns:
        results:
            An instance of the Results class containing the results
            of your Query.

    Example:
        >>> # filter of "NSA redshift less than 0.1 and IFU names starting with 19"
        >>> searchfilter = 'nsa_redshift < 0.1 and ifu.name = 19*'
        >>> returnparams = ['cube.ra', 'cube.dec']
        >>> q = Query(searchfilter=searchfilter, returnparams=returnparams)
        >>> results = q.run()

    '''

    def __init__(self, *args, **kwargs):

        self.query = None
        self.params = []
        self.filterparams = {}
        self.queryparams = None
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
        returnparams = kwargs.get('returnparams', None)
        if returnparams:
            self.set_returnparams(returnparams)

        # set default parameters
        self.set_defaultparams()

        # if searchfilter is set then set the parameters
        searchfilter = kwargs.get('searchfilter', None)
        if searchfilter:
            self.set_filter(searchfilter=searchfilter)

        # Don't do anything if nothing specified
        allnot = [not searchfilter, not returnparams]
        print('inside query', allnot, not all(allnot), self.mode, not all(allnot) and self.mode == 'local')
        if not all(allnot) and self.mode == 'local':
            # create query parameter ModelClasses
            self._create_query_modelclasses()

            # join tables
            self._join_tables()

            # add condition
            if searchfilter:
                self.add_condition()

            # add PipelineInfo
            self._addPipeline()

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

    def set_returnparams(self, returnparams):
        ''' Loads the user input parameters into the query params limit
        '''
        self.params.extend(returnparams)

    def set_defaultparams(self):
        ''' Loads the default params for a given return type '''
        # TODO - change mangaid to plateifu once plateifu works in
        # SQLalchemy_boolean_search and we can figure out how to grab the classes
        # for hybrid properties
        assert self.returntype in [None, 'cube', 'spaxel', 'map', 'rss'], 'Query returntype must be either cube, spaxel, map, rss'
        self.defaultparams = ['cube.mangaid', 'cube.plate', 'ifu.name']  # cube.plate,ifu.name temp until cube.plateifu works
        if self.returntype == 'spaxel':
            self.defaultparams.extend(['spaxel.x', 'spaxel.y'])
        elif self.returntype == 'rssfiber':
            self.defaultparams.extend(['rssfiber.fiber.fiberid'])
        elif self.returntype == 'map':
            pass

        # add to main set of params
        self.params.extend(self.defaultparams)

    def _create_query_modelclasses(self):
        ''' Creates a list of database ModelClasses from a list of parameter names '''
        self.params = [item for item in self.params if item in set(self.params)]
        print('my params', self.params)
        self.queryparams = self.marvinform._param_form_lookup.mapToColumn(self.params)
        self.queryparams = [item for item in self.queryparams if item in set(self.queryparams)]
        self.queryparams_order = [q.key for q in self.queryparams]
        print('queryorder', self.queryparams_order)

    def set_filter(self, searchfilter=None):
        ''' Parses a filter string and adds it into the query.

        Parses a natural language string filter into the appropriate SQL
        filter syntax.  String is a boolean join of one or more conditons
        of the form "PARAMETER_NAME OPERAND VALUE"

        Parameter names must be uniquely specified. For example, nsa_redshift is
        a unique parameter name in the database and can be specified thusly.
        On the other hand, name is not a unique parameter name in the database,
        and must be clarified with the desired table.

        Parameter Naming Convention:
            NSA redshift == nsa_redshift
            IFU name == ifu.name
            Pipeline name == pipeline_info.name

        Allowed Joins:
            AND | OR | NOT

            In the absence of parantheses, the precedence of
            joins follow: NOT > AND > OR

        Allowed Operands:
            == | != | <= | >= | < | > | =

            Notes:
                Operand == maps to a strict equality (x == 5 --> x is equal to 5)

                Operand = maps to SQL LIKE

                (x = 5 --> x contains the string 5; x = '%5%')

                (x = 5* --> x starts with the string 5; x = '5%')

                (x = *5 --> x ends with the string 5; x = '%5')

        Parameters:
            searchfilter (str):
                A (natural language) string containing the filter conditions
                in the query; written as you would say it.

        Example:
            >>> # Filter string
            >>> filter = "nsa_redshift < 0.012 and ifu.name = 19*"
            >>> # Converts to
            >>> and_(nsa_redshift<0.012, ifu.name=19*)
            >>> # SQL syntax
            >>> mangadatadb.sample.nsa_redshift < 0.012 AND lower(mangadatadb.ifudesign.name) LIKE lower('19%')

            >>> # Filter string
            >>> filter = 'cube.plate < 8000 and ifu.name = 19 or not (nsa_redshift > 0.1 or not cube.ra > 225.)'
            >>> # Converts to
            >>> or_(and_(cube.plate<8000, ifu.name=19), not_(or_(nsa_redshift>0.1, not_(cube.ra>225.))))
            >>> # SQL syntax
            >>> mangadatadb.cube.plate < 8000 AND lower(mangadatadb.ifudesign.name) LIKE lower(('%' || '19' || '%'))
            >>> OR NOT (mangadatadb.sample.nsa_redshift > 0.1 OR mangadatadb.cube.ra <= 225.0)
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
        self._modellist = [param.class_ for param in self.queryparams]

        # Gets the list of joins from ModelGraph. Uses Cube as nexus, so that
        # the order of the joins is the correct one.
        # TODO: at some point, all the queries should be generalised so that
        # we don't assume that we are querying a cube.
        joinmodellist = self._modelgraph.getJoins(self._modellist, format_out='models', nexus=marvindb.datadb.Cube)

        # sublist = [model for model in modellist if model.__tablename__ not in self._basetable and not self._tableInQuery(model.__tablename__)]
        # self.joins.extend([model.__tablename__ for model in sublist])
        # self.query = self.query.join(*sublist)
        for model in joinmodellist:
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

        infilter = None
        if names:
            if not isinstance(self.query, type(None)):
                if not isinstance(self.query.whereclause, type(None)):
                    wc = str(self.query.whereclause.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))
                    infilter = any([name in wc for name in names])

        return infilter

    @makeBaseQuery
    @checkCondition
    @updateConfig
    def run(self, qmode='all'):
        ''' Runs a Marvin Query

            Runs the query and return an instance of Marvin Results class
            to deal with results.  Input qmode allows to perform
            different sqlalchemy queries

            Parameters:
                qmode ({'all', 'one', 'first', 'count'}):
                    String indicating

            Returns:
                results (object):
                    An instance of the Marvin Results class containing the
                    results from the Query.

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

            return Results(results=res, query=self.query, count=count, mode=self.mode, returntype=self.returntype, queryobj=self)

        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['querycubes']['url']

            params = {'searchfilter': self.searchfilter, 'params': self.params}
            try:
                ii = Interaction(route=url, params=params)
            except MarvinError as e:
                raise MarvinError('API Query call failed: {0}'.format(e))
            else:
                res = ii.getData()
                self.queryparams_order = ii.results['queryparams_order']
                self.query = ii.results['query']
            return Results(results=res, query=self.query, mode=self.mode, queryobj=self, count=len(res), returntype=self.returntype)

    def _sortQuery(self):
        ''' Sort the query by a given parameter '''

        if not isinstance(self.sort, type(None)):
            # set the sort variable ModelClass parameter
            sortparam = self.marvinform._param_form_lookup.mapToColumn(self.sort)

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
            print('Cannot show full SQL query in remote mode, use the Results showQuery')

    def reset(self):
        ''' Resets all query attributes '''
        self.__init__()

    @updateConfig
    def _createBaseQuery(self):
        ''' Create the base query session object.  Passes in a list of parameters defined in
            returnparams, filterparams, and defaultparams
        '''
        self.query = self.session.query(*self.queryparams)

    def _getPipeInfo(self, pipename):
        ''' Retrieve the pipeline Info for a given pipeline version name '''

        assert pipename.lower() in ['drp', 'dap'], 'Pipeline Name must either be DRP or DAP'

        # bindparam values
        bindname = 'drpver' if pipename.lower() == 'drp' else 'dapver'
        bindvalue = config.drpver if pipename.lower() == 'drp' else config.dapver

        # class names
        if pipename.lower() == 'drp':
            inclasses = self._tableInQuery('cube') or 'cube' in str(self.query.statement.compile())
        elif pipename.lower() == 'dap':
            inclasses = self._tableInQuery('file') or 'file' in str(self.query.statement.compile())

        # set alias
        pipealias = self._drp_alias if pipename.lower() == 'drp' else self._dap_alias

        # get the pipeinfo
        if inclasses:
            pipeinfo = marvindb.session.query(pipealias).\
                join(marvindb.datadb.PipelineName, marvindb.datadb.PipelineVersion).\
                filter(marvindb.datadb.PipelineName.label == pipename.upper(),
                       marvindb.datadb.PipelineVersion.version == bindparam(bindname, bindvalue)).one()
        else:
            pipeinfo = None

        return pipeinfo

    def _addPipeline(self):
        ''' Adds the DRP and DAP Pipeline Info into the Query '''

        self._drp_alias = aliased(marvindb.datadb.PipelineInfo, name='drpalias')
        self._dap_alias = aliased(marvindb.datadb.PipelineInfo, name='dapalias')

        drppipe = self._getPipeInfo('drp')
        dappipe = self._getPipeInfo('dap')

        # Add DRP pipeline version
        if drppipe:
            self.query = self.query.join(self._drp_alias, marvindb.datadb.Cube.pipelineInfo).\
                filter(self._drp_alias.pk == drppipe.pk)

        # Add DAP pipeline version
        if dappipe:
            self.query = self.query.join(self._dap_alias, marvindb.dapdb.File.pipelineinfo).\
                filter(self._dap_alias.pk == dappipe.pk)

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
