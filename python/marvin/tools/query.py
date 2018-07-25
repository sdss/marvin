#!/usr/bin/env python
# encoding: utf-8

# Licensed under a 3-clause BSD license.

# Revision History:
#     Initial Version: 2016-02-17 14:13:28 by Brett Andrews
#     2016-02-23 - Modified to test a programmatic query using a test sample form - B. Cherinka
#     2016-03-02 - Generalized to many parameters and many forms - B. Cherinka
#                - Added config drpver info
#     2016-03-12 - Changed parameter input to be a natural language string

from __future__ import division, print_function, unicode_literals

import datetime
import os
import re
import warnings
from collections import OrderedDict, defaultdict
from functools import wraps
from operator import eq, ge, gt, le, lt, ne

import numpy as np
import six
from marvin import config, marvindb
from marvin.api.api import Interaction
from marvin.core import marvin_pickle
from marvin.core.exceptions import (MarvinBreadCrumb, MarvinError, MarvinUserWarning)
from marvin.tools.results import Results, remote_mode_only
from marvin.utils.datamodel.query import datamodel
from marvin.utils.datamodel.query.base import query_params
from marvin.utils.general import temp_setattr
from marvin.utils.general.structs import string_folding_wrapper
from sqlalchemy import bindparam, func
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import aliased
from sqlalchemy.sql.expression import desc
from sqlalchemy_boolean_search import (BooleanSearchException, parse_boolean_search)

try:
    import cPickle as pickle
except:
    import pickle

__all__ = ['Query', 'doQuery']
opdict = {'<=': le, '>=': ge, '>': gt, '<': lt, '!=': ne, '=': eq, '==': eq}

breadcrumb = MarvinBreadCrumb()


def tree():
    return defaultdict(tree)


def doQuery(*args, **kwargs):
    """Convenience function for building a Query and retrieving the Results.

    Parameters:
        N/A:
            See the :class:`~marvin.tools.query.Query` class for a list
            of inputs.

    Returns:
        query, results:
            A tuple containing the built
            :class:`~marvin.tools.query.Query` instance, and the
            :class:`~marvin.tools.results.Results` instance.
    """
    start = kwargs.pop('start', None)
    end = kwargs.pop('end', None)
    q = Query(*args, **kwargs)
    try:
        res = q.run(start=start, end=end)
    except TypeError as e:
        warnings.warn('Cannot run, query object is None: {0}.'.format(e), MarvinUserWarning)
        res = None

    return q, res


def updateConfig(f):
    """Decorator that updates query object with new config drpver version."""

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if self.query and self.mode == 'local':
            self.query = self.query.params({'drpver': self._drpver, 'dapver': self._dapver})
        return f(self, *args, **kwargs)
    return wrapper


def makeBaseQuery(f):
    """Decorator that makes the base query if it does not already exist."""

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.query and self.mode == 'local':
            self._createBaseQuery()
        return f(self, *args, **kwargs)
    return wrapper


def checkCondition(f):
    """Decorator that checks if filter is set, if it does not already exist."""

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
            An instance of the :class:`~marvin.tools.query.results.Results`
            class containing the results of your Query.

    Example:
        >>> # filter of "NSA redshift less than 0.1 and IFU names starting with 19"
        >>> searchfilter = 'nsa.z < 0.1 and ifu.name = 19*'
        >>> returnparams = ['cube.ra', 'cube.dec']
        >>> q = Query(searchfilter=searchfilter, returnparams=returnparams)
        >>> results = q.run()

    '''

    def __init__(self, *args, **kwargs):

        self._release = kwargs.pop('release', config.release)
        self._drpver, self._dapver = config.lookUpVersions(release=self._release)

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
        self.quiet = kwargs.get('quiet', None)
        self._errors = []
        self._basetable = None
        self._modelgraph = marvindb.modelgraph
        self._returnparams = []
        self._caching = kwargs.get('caching', True)
        self.verbose = kwargs.get('verbose', True)
        self.count_threshold = kwargs.get('count_threshold', 1000)
        self.allspaxels = kwargs.get('allspaxels', None)
        self.mode = kwargs.get('mode', None)
        self.limit = int(kwargs.get('limit', 100))
        self.sort = kwargs.get('sort', 'mangaid')
        self.order = kwargs.get('order', 'asc')
        self.return_all = kwargs.get('return_all', False)
        self.datamodel = datamodel[self._release]
        self.marvinform = self.datamodel._marvinform

        # drop breadcrumb
        breadcrumb.drop(message='Initializing MarvinQuery {0}'.format(self.__class__),
                        category=self.__class__)

        # set the mode
        if self.mode is None:
            self.mode = config.mode

        if self.mode == 'local':
            self._doLocal()
        if self.mode == 'remote':
            self._doRemote()
        if self.mode == 'auto':
            try:
                self._doLocal()
            except Exception as e:
                warnings.warn('local mode failed. Trying remote now.', MarvinUserWarning)
                self._doRemote()

        # get return type
        self.returntype = kwargs.get('returntype', None)

        # set default parameters
        self.set_defaultparams()

        # get user-defined input parameters
        returnparams = kwargs.get('returnparams', [])
        if returnparams:
            self.set_returnparams(returnparams)

        # if searchfilter is set then set the parameters
        searchfilter = kwargs.get('searchfilter', None)
        if searchfilter:
            self.set_filter(searchfilter=searchfilter)
            self._isdapquery = self._checkInFilter(name='dapdb')

        # Don't do anything if nothing specified
        allnot = [not searchfilter, not returnparams]
        if not all(allnot) and self.mode == 'local':
            # create query parameter ModelClasses
            self._create_query_modelclasses()
            # this adds spaxel x, y into default for query 1 dap zonal query
            self._adjust_defaults()

            # join tables
            self._join_tables()

            # add condition
            if searchfilter:
                self.add_condition()

            # add PipelineInfo
            self._addPipeline()

            # check if query if a dap query
            if self._isdapquery:
                self._buildDapQuery()
                self._check_dapall_query()

    def __repr__(self):
        return ('Marvin Query(filter={4}, mode={0}, limit={1}, sort={2}, order={3})'
                .format(repr(self.mode), self.limit, self.sort, repr(self.order), self.searchfilter))

    def _doLocal(self):
        ''' Tests if it is possible to perform queries locally. '''

        if not config.db or not self.session:
            warnings.warn('No local database found. Cannot perform queries.', MarvinUserWarning)
            raise MarvinError('No local database found.  Query cannot be run in local mode')
        else:
            self.mode = 'local'

    def _doRemote(self):
        ''' Sets up to perform queries remotely. '''

        if not config.urlmap:
            raise MarvinError('No URL Map found.  Cannot make remote query calls!')
        else:
            self.mode = 'remote'

    def _check_query(self, name):
        ''' Check if string is inside the query statement '''

        qstate = str(self.query.statement.compile(compile_kwargs={'literal_binds':True}))
        return name in qstate

    def _checkInFilter(self, name='dapdb'):
        ''' Check if the given name is in the schema of any of the filter params '''

        if self.mode == 'local':
            fparams = self.marvinform._param_form_lookup.mapToColumn(self.filterparams.keys())
            fparams = [fparams] if not isinstance(fparams, list) else fparams
            inschema = [name in c.class_.__table__.schema for c in fparams]
        elif self.mode == 'remote':
            inschema = []
        return True if any(inschema) else False

    def _check_shortcuts_in_filter(self, strfilter):
        ''' Check for shortcuts in string filter

            Replaces shortcuts in string searchfilter
            with the full tables and names.

            is there a better way?
        '''
        # table shortcuts
        # for key in self.marvinform._param_form_lookup._tableShortcuts.keys():
        #     #if key in strfilter:
        #     if re.search('{0}.[a-z]'.format(key), strfilter):
        #         strfilter = strfilter.replace(key, self.marvinform._param_form_lookup._tableShortcuts[key])

        # name shortcuts
        for key in self.marvinform._param_form_lookup._nameShortcuts.keys():
            if key in strfilter:
                # strfilter = strfilter.replace(key, self.marvinform._param_form_lookup._nameShortcuts[key])
                param_form_lookup = self.marvinform._param_form_lookup
                strfilter = re.sub(r'\b{0}\b'.format(key),
                                   '{0}'.format(param_form_lookup._nameShortcuts[key]),
                                   strfilter)
        return strfilter

    def _adjust_defaults(self):
        ''' Adjust the default parameters to include necessary parameters

        For any query involving DAP DB, always return the spaxel index
        TODO: change this to spaxel x and y

        TODO: change this entirely

        '''
        dapschema = ['dapdb' in c.class_.__table__.schema for c in self.queryparams]
        if any(dapschema):
            dapcols = ['spaxelprop.x', 'spaxelprop.y', 'bintype.name', 'template.name']
            self.defaultparams.extend(dapcols)
            self.params.extend(dapcols)
            self.params = list(OrderedDict.fromkeys(self.params))
            self._create_query_modelclasses()
            # qpdap = self.marvinform._param_form_lookup.mapToColumn(dapcols)
            # self.queryparams.extend(qpdap)
            # self.queryparams_order.extend([q.key for q in qpdap])

    def set_returnparams(self, returnparams):
        ''' Loads the user input parameters into the query params limit

        Adds a list of string parameter names into the main list of
        query parameters to return

        Parameters:
            returnparams (list):
                A string list of the parameters you wish to return in the query

        '''
        if returnparams:
            returnparams = [returnparams] if not isinstance(returnparams, list) else returnparams

            # look up shortcut names for the return parameters
            full_returnparams = [self.marvinform._param_form_lookup._nameShortcuts[rp]
                                 if rp in self.marvinform._param_form_lookup._nameShortcuts else rp
                                 for rp in returnparams]

            self._returnparams = full_returnparams
        self.params.extend(full_returnparams)

    def set_defaultparams(self):
        ''' Loads the default params for a given return type
        TODO - change mangaid to plateifu once plateifu works in

        cube, maps, rss, modelcube - file objects
        spaxel, map, rssfiber - derived objects (no file)

        these are also the default params except
        any query on spaxelprop should return spaxel_index (x/y)

        Minimum parameters to instantiate a Marvin Tool
        cube - return plateifu/mangaid
        modelcube - return plateifu/mangaid, bintype, template
        rss - return plateifu/mangaid
        maps - return plateifu/mangaid, bintype, template
        spaxel - return plateifu/mangaid, spaxel x and y

        map - do not instantiate directly (plateifu/mangaid, bintype, template, property name, channel)
        rssfiber - do not instantiate directly (plateifu/mangaid, fiberid)

        return any of our tools
        '''
        assert self.returntype in [None, 'cube', 'spaxel', 'maps',
                                   'rss', 'modelcube'], 'Query returntype must be either cube, spaxel, maps, modelcube, rss'
        self.defaultparams = ['cube.mangaid', 'cube.plate', 'cube.plateifu', 'ifu.name']
        if self.returntype == 'spaxel':
            pass
            #self.defaultparams.extend(['spaxel.x', 'spaxel.y'])
        elif self.returntype == 'modelcube':
            self.defaultparams.extend(['bintype.name', 'template.name'])
        elif self.returntype == 'rss':
            pass
        elif self.returntype == 'maps':
            self.defaultparams.extend(['bintype.name', 'template.name'])
            # self.defaultparams.extend(['spaxelprop.x', 'spaxelprop.y'])

        # add to main set of params
        self.params.extend(self.defaultparams)

    def _create_query_modelclasses(self):
        ''' Creates a list of database ModelClasses from a list of parameter names '''
        self.params = [item for item in self.params if item in set(self.params)]
        self.queryparams = self.marvinform._param_form_lookup.mapToColumn(self.params)
        self.queryparams = [item for item in self.queryparams if item in set(self.queryparams)]
        self.queryparams_order = [q.key for q in self.queryparams]

    def get_available_params(self, paramdisplay='best'):
        ''' Retrieve the available parameters to query on

        Retrieves a list of the available query parameters.
        Can either retrieve a list of all the parameters or only the vetted parameters.

        Parameters:
            paramdisplay (str {all|best}):
                String indicating to grab either all or just the vetted parameters.
                Default is to only return 'best', i.e. vetted parameters

        Returns:
            qparams (list):
                a list of all of the available queryable parameters
        '''
        assert paramdisplay in ['all', 'best'], 'paramdisplay can only be either "all" or "best"!'

        if paramdisplay == 'all':
            qparams = self.datamodel.groups.list_params('full')
        elif paramdisplay == 'best':
            qparams = query_params
        return qparams

    @remote_mode_only
    def save(self, path=None, overwrite=False):
        ''' Save the query as a pickle object

        Parameters:
            path (str):
                Filepath and name of the pickled object
            overwrite (bool):
                Set this to overwrite an existing pickled file

        Returns:
            path (str):
                The filepath and name of the pickled object

        '''

        sf = self.searchfilter.replace(' ', '') if self.searchfilter else 'anon'
        # set the path
        if not path:
            path = os.path.expanduser('~/marvin_query_{0}.mpf'.format(sf))

        # check for file extension
        if not os.path.splitext(path)[1]:
            path = os.path.join(path + '.mpf')

        path = os.path.realpath(path)

        if os.path.isdir(path):
            raise MarvinError('path must be a full route, including the filename.')

        if os.path.exists(path) and not overwrite:
            warnings.warn('file already exists. Not overwriting.', MarvinUserWarning)
            return

        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # set bad pickled attributes to None
        attrs = ['session', 'datamodel', 'marvinform', 'myform', '_modelgraph']

        # pickle the query
        try:
            with temp_setattr(self, attrs, None):
                pickle.dump(self, open(path, 'wb'), protocol=-1)
        except Exception as ee:
            if os.path.exists(path):
                os.remove(path)
            raise MarvinError('Error found while pickling: {0}'.format(str(ee)))

        return path

    @classmethod
    def restore(cls, path, delete=False):
        ''' Restore a pickled object

        Parameters:
            path (str):
                The filename and path to the pickled object

            delete (bool):
                Turn this on to delete the pickled fil upon restore

        Returns:
            Query (instance):
                The instantiated Marvin Query class
        '''
        obj = marvin_pickle.restore(path, delete=delete)
        obj._modelgraph = marvindb.modelgraph
        obj.session = marvindb.session
        obj.datamodel = datamodel[obj._release]
        # if obj.allspaxels:
        #     obj.datamodel.use_all_spaxels()
        obj.marvinform = obj.datamodel._marvinform
        return obj

    def set_filter(self, searchfilter=None):
        ''' Parses a filter string and adds it into the query.

        Parses a natural language string filter into the appropriate SQL
        filter syntax.  String is a boolean join of one or more conditons
        of the form "PARAMETER_NAME OPERAND VALUE"

        Parameter names must be uniquely specified. For example, nsa.z is
        a unique parameter name in the database and can be specified thusly.
        On the other hand, name is not a unique parameter name in the database,
        and must be clarified with the desired table.

        Parameter Naming Convention:
            NSA redshift == nsa.z
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
            >>> filter = "nsa.z < 0.012 and ifu.name = 19*"
            >>> # Converts to
            >>> and_(nsa.z<0.012, ifu.name=19*)
            >>> # SQL syntax
            >>> mangasampledb.nsa.z < 0.012 AND lower(mangadatadb.ifudesign.name) LIKE lower('19%')

            >>> # Filter string
            >>> filter = 'cube.plate < 8000 and ifu.name = 19 or not (nsa.z > 0.1 or not cube.ra > 225.)'
            >>> # Converts to
            >>> or_(and_(cube.plate<8000, ifu.name=19), not_(or_(nsa.z>0.1, not_(cube.ra>225.))))
            >>> # SQL syntax
            >>> mangadatadb.cube.plate < 8000 AND lower(mangadatadb.ifudesign.name) LIKE lower(('%' || '19' || '%'))
            >>> OR NOT (mangasampledb.nsa.z > 0.1 OR mangadatadb.cube.ra <= 225.0)
        '''

        if searchfilter:
            # if params is a string, then parse and filter
            if isinstance(searchfilter, six.string_types):
                searchfilter = self._check_shortcuts_in_filter(searchfilter)
                try:
                    parsed = parse_boolean_search(searchfilter)
                except BooleanSearchException as e:
                    raise MarvinError('Your boolean expression contained a syntax error: {0}'.format(e))
            else:
                raise MarvinError('Input parameters must be a natural language string!')

            # update the parameters dictionary
            self.searchfilter = searchfilter
            self._parsed = parsed
            self._checkParsed()
            self.strfilter = str(parsed)
            self.filterparams.update(parsed.params)
            filterkeys = [key for key in parsed.uniqueparams if key not in self.params]
            self.params.extend(filterkeys)

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

        formkeys = list(self.myforms.keys())
        isgood = [form.validate() for form in self.myforms.values()]
        if not all(isgood):
            inds = np.where(np.invert(isgood))[0]
            for index in inds:
                self._errors.append(list(self.myforms.values())[index].errors)
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
            name = '{0}.{1}'.format(model.__table__.schema, model.__tablename__)
            if not self._tableInQuery(name):
                self.joins.append(model.__tablename__)
                if 'template' not in model.__tablename__:
                    self.query = self.query.join(model)
                else:
                    # assume template_kin only now, TODO deal with template_pop later
                    self.query = self.query.join(model, marvindb.dapdb.Structure.template_kin)

    def build_filter(self):
        ''' Builds a filter condition to load into sqlalchemy filter. '''
        try:
            self.filter = self._parsed.filter(self._modellist)
        except BooleanSearchException as e:
            raise MarvinError('Your boolean expression could not me mapped to model: {0}'.format(e))

    def update_params(self, param):
        ''' Update the input parameters '''
        # param = {key: unicode(val) if '*' not in unicode(val) else unicode(val.replace('*', '%')) for key, val in param.items() if key in self.filterparams.keys()}
        param = {key: val.decode('UTF-8') if '*' not in val.decode('UTF-8') else val.replace('*', '%').decode('UTF-8') for key, val in param.items() if key in self.filterparams.keys()}
        self.filterparams.update(param)
        self._setForms()

    def _update_params(self, param):
        ''' this is now broken, this should update the boolean params in the filter condition '''

        ''' Update any input parameters that have been bound already.  Input is a dictionary of key, value pairs representing
            parameter name to update, and the value (number only) to update.  This does not allow to change the operand.
            Does not update self.params
            e.g.
            original input parameters {'nsa.z': '< 0.012'}
            newparams = {'nsa.z': '0.2'}
            update_params(newparams)
            new condition will be nsa.z < 0.2
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
    def run(self, start=None, end=None, raw=None, orm=None, core=None):
        ''' Runs a Marvin Query

            Runs the query and return an instance of Marvin Results class
            to deal with results.

            Parameters:
                start (int):
                    Starting value of a subset.  Default is None
                end (int):
                    Ending value of a subset.  Default is None

            Returns:
                results (object):
                    An instance of the Marvin Results class containing the
                    results from the Query.

        '''
        if self.mode == 'local':

            # Check for adding a sort
            self._sortQuery()

            # Check to add the cache
            if self._caching:
                from marvin.core.caching_query import FromCache
                self.query = self.query.options(FromCache("default")).\
                    options(*marvindb.cache_bits)

            # turn on streaming of results
            self.query = self.query.execution_options(stream_results=True)

            # get total count, and if more than 150 results, paginate and only return the first 100
            starttime = datetime.datetime.now()

            # check for query and get count
            if marvindb.isdbconnected:
                qm = self._check_history(check_only=True)
                self.totalcount = qm.count if qm else None

            # run count if it doesn't exist
            if self.totalcount is None:
                self.totalcount = self.query.count()

            # get the new count if start and end exist
            if start and end:
                count = (end - start)
            else:
                count = self.totalcount

            # # run the query
            # res = self.query.slice(start, end).all()
            # count = len(res)
            # self.totalcount = count if not self.totalcount else self.totalcount

            # check history
            if marvindb.isdbconnected:
                query_meta = self._check_history()

            if count > self.count_threshold and self.return_all is False:
                # res = res[0:self.limit]
                start = 0
                end = self.limit
                count = (end - start)
                warnings.warn('Results contain more than {0} entries.  '
                              'Only returning first {1}'.format(self.count_threshold, self.limit), MarvinUserWarning)
            elif self.return_all is True:
                warnings.warn('Warning: Attempting to return all results. This may take a long time or crash.', MarvinUserWarning)
                start = None
                end = None
            elif start and end:
                warnings.warn('Getting subset of data {0} to {1}'.format(start, end), MarvinUserWarning)

            # slice the query
            query = self.query.slice(start, end)

            # run the query
            if not any([raw, core, orm]):
                raw = True

            if raw:
                # use the db api cursor
                sql = str(self._get_sql(query))
                conn = marvindb.db.engine.raw_connection()
                cursor = conn.cursor('query_cursor')
                cursor.execute(sql)
                res = self._fetch_data(cursor)
                conn.close()
            elif core:
                # use the core connection
                sql = str(self._get_sql(query))
                with marvindb.db.engine.connect() as conn:
                    results = conn.execution_options(stream_results=True).execute(sql)
                    res = self._fetch_data(results)
            elif orm:
                # use the orm query
                yield_num = int(10**(np.floor(np.log10(self.totalcount))))
                results = string_folding_wrapper(query.yield_per(yield_num), keys=self.params)
                res = list(results)

            # get the runtime
            endtime = datetime.datetime.now()
            self.runtime = (endtime - starttime)

            # clear the session
            self.session.close()

            # pass the results into Marvin Results
            final = Results(results=res, query=query, count=count, mode=self.mode,
                            returntype=self.returntype, queryobj=self, totalcount=self.totalcount,
                            chunk=self.limit, runtime=self.runtime, start=start, end=end)

            # get the final time
            posttime = datetime.datetime.now()
            self.finaltime = (posttime - starttime)

            return final

        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            if self.return_all:
                warnings.warn('Warning: Attempting to return all results. This may take a long time or crash.')

            # Get the query route
            url = config.urlmap['api']['querycubes']['url']

            params = {'searchfilter': self.searchfilter,
                      'params': ','.join(self._returnparams) if self._returnparams else None,
                      'returntype': self.returntype,
                      'limit': self.limit,
                      'sort': self.sort, 'order': self.order,
                      'release': self._release,
                      'return_all': self.return_all,
                      'start': start,
                      'end': end,
                      'caching': self._caching}
            try:
                ii = Interaction(route=url, params=params, stream=True)
            except Exception as e:
                # if a remote query fails for any reason, then try to clean them up
                # self._cleanUpQueries()
                raise MarvinError('API Query call failed: {0}'.format(e))
            else:
                res = ii.getData()
                self.queryparams_order = ii.results['queryparams_order']
                self.params = ii.results['params']
                self.query = ii.results['query']
                count = ii.results['count']
                chunk = int(ii.results['chunk'])
                totalcount = ii.results['totalcount']
                query_runtime = ii.results['runtime']
                resp_runtime = ii.response_time

            if self.return_all:
                msg = 'Returning all {0} results'.format(totalcount)
            else:
                msg = 'Only returning the first {0} results.'.format(count)

            if not self.quiet:
                print('Results contain of a total of {0}. {1}'.format(totalcount, msg))
            return Results(results=res, query=self.query, mode=self.mode, queryobj=self, count=count,
                           returntype=self.returntype, totalcount=totalcount, chunk=chunk,
                           runtime=query_runtime, response_time=resp_runtime, start=start, end=end)

    def _fetch_data(self, obj):
        ''' Fetch query using fetchall or fetchmany '''

        res = []

        if not self.return_all:
            res = obj.fetchall()
        else:
            while True:
                rows = obj.fetchmany(100000)
                if rows:
                    res.extend(rows)
                else:
                    break
        return res

    def _check_history(self, check_only=None):
        ''' Check the query against the query history schema '''

        sqlcol = self.marvinform._param_form_lookup.mapToColumn('sql')
        stringfilter = self.searchfilter.strip().replace(' ', '')
        rawsql = self.show().strip()
        return_params = ','.join(self._returnparams)
        qm = self.session.query(sqlcol.class_).\
            filter(sqlcol == rawsql, sqlcol.class_.release == self._release).one_or_none()

        if check_only:
            return qm

        with self.session.begin():
            if not qm:
                qm = sqlcol.class_(searchfilter=stringfilter, n_run=1, release=self._release,
                                   count=self.totalcount, sql=rawsql, return_params=return_params)
                self.session.add(qm)
            else:
                qm.n_run += 1

        return qm

    def _cleanUpQueries(self):
        ''' Attempt to clean up idle queries on the server

        This is a hack to try to kill all idl processes on the server.
        Using pg_terminate_backend and pg_stat_activity it terminates all
        transactions that are in an idle, or idle in transaction, state
        that have running for > 1 minute, and whose application_name is
        not psql, and the process is not the one initiating the terminate.

        The rank part ranks the processes and originally killed all > 1, to
        leave one alive as a warning to the others.  I've changed this to 0
        to kill everything.

        I think this will sometimes also leave a newly orphaned idle
        ROLLBACK transaction.  Not sure why.

        '''
        if self.mode == 'local':
            sql = ("with inactive as (select p.pid, rank() over (partition by \
                   p.client_addr order by p.backend_start ASC) as rank from \
                   pg_stat_activity as p where p.application_name !~ 'psql' \
                   and p.state ilike '%idle%' and p.pid <> pg_backend_pid() and \
                   current_timestamp-p.state_change > interval '1 minutes') \
                   select pg_terminate_backend(pid) from inactive where rank > 0;")
            self.session.expire_all()
            self.session.expunge_all()
            res = self.session.execute(sql)
            tmp = res.fetchall()
            #self.session.close()
            #marvindb.db.engine.dispose()
        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['cleanupqueries']['url']

            params = {'task': 'clean', 'release': self._release}

            try:
                ii = Interaction(route=url, params=params)
            except Exception as e:
                raise MarvinError('API Query call failed: {0}'.format(e))
            else:
                res = ii.getData()

    def _getIdleProcesses(self):
        ''' Get a list of all idle processes on server

        This grabs a list of all processes in a state of
        idle, or idle in transaction using pg_stat_activity
        and returns the process id, the state, and the query

        '''
        if self.mode == 'local':
            sql = ("select p.pid,p.state,p.query from pg_stat_activity as p \
                   where p.state ilike '%idle%';")
            res = self.session.execute(sql)
            procs = res.fetchall()
        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['cleanupqueries']['url']

            params = {'task': 'getprocs', 'release': self._release}

            try:
                ii = Interaction(route=url, params=params)
            except Exception as e:
                raise MarvinError('API Query call failed: {0}'.format(e))
            else:
                procs = ii.getData()
        return procs

    def _sortQuery(self):
        ''' Sort the query by a given parameter '''

        if not isinstance(self.sort, type(None)):
            # set the sort variable ModelClass parameter
            if '.' in self.sort:
                param = self.datamodel.parameters[str(self.sort)].full
            else:
                param = self.datamodel.parameters.get_full_from_remote(self.sort)
            sortparam = self.marvinform._param_form_lookup.mapToColumn(param)

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
        ''' Prints into to the console

        Displays the query to the console with parameter variables plugged in.
        Works only in local mode.  Input prop can be one of Can be one of query,
        tables, joins, or filter.

        Only works in LOCAL mode.

        Allowed Values for Prop:
            query - displays the entire query (default if nothing specified)
            tables - displays the tables that have been joined in the query
            joins -  same as table
            filter - displays only the filter used on the query

        Parameters:
            prop (str):
                The type of info to print.

        Example:
            TODO add example

        '''

        assert prop in [None, 'query', 'tables', 'joins', 'filter'], 'Input must be query, tables, joins, or filter'

        if self.mode == 'local':
            if not prop or 'query' in prop:
                sql = self._get_sql(self.query)
            elif prop == 'tables':
                sql = self.joins
            elif prop == 'filter':
                '''oddly this does not update when bound parameters change, but the statement above does '''
                sql = self.query.whereclause.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True})
            else:
                sql = self.__getattribute__(prop)

            return str(sql)
        elif self.mode == 'remote':
            sql = 'Cannot show full SQL query in remote mode, use the Results showQuery'
            warnings.warn(sql, MarvinUserWarning)
            return sql

    def _get_sql(self, query):
        ''' Get the sql for a given query

        Parameters:
            query (object):
                An SQLAlchemy Query object

        Returms:
            A raw sql string
        '''

        return query.statement.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True})

    def reset(self):
        ''' Resets all query attributes '''
        self.__init__()

    @updateConfig
    def _createBaseQuery(self):
        ''' Create the base query session object.  Passes in a list of parameters defined in
            returnparams, filterparams, and defaultparams
        '''
        labeledqps = [qp.label(self.params[i]) for i, qp in enumerate(self.queryparams)]
        self.query = self.session.query(*labeledqps)

    def _query_column(self, column_name):
        ''' query and return a specific column from the current query '''
        qp = self.marvinform._param_form_lookup.mapToColumn(column_name)
        qp = qp.label(column_name)
        return self.query.from_self(qp).all()

    def _getPipeInfo(self, pipename):
        ''' Retrieve the pipeline Info for a given pipeline version name '''

        assert pipename.lower() in ['drp', 'dap'], 'Pipeline Name must either be DRP or DAP'

        # bindparam values
        bindname = 'drpver' if pipename.lower() == 'drp' else 'dapver'
        bindvalue = self._drpver if pipename.lower() == 'drp' else self._dapver

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
            if isinstance(self.query, six.string_types):
                isin = name in self.query
            else:
                isin = False
        return isin

    def _group_by(self, params=None):
        ''' Group the query by a set of parameters

        Parameters:
            params (list):
                A list of string parameter names to group the query by

        Returns:
            A new SQLA Query object
        '''

        if not params:
            params = [d for d in self.defaultparams if 'spaxelprop' not in d]

        newdefaults = self.marvinform._param_form_lookup.mapToColumn(params)
        self.params = params
        newq = self.query.from_self(*newdefaults).group_by(*newdefaults)
        return newq

    # ------------------------------------------------------
    #  DAP Specific Query Modifiers - subqueries, etc go below here
    #  -----------------------------------------------------

    def _buildDapQuery(self):
        ''' Builds a DAP zonal query
        '''

        # get the appropriate Junk (SpaxelProp) ModelClass
        self._junkclass = self.marvinform.\
            _param_form_lookup['spaxelprop.file'].Meta.model

        # get good spaxels
        # bingood = self.getGoodSpaxels()
        # self.query = self.query.\
        #     join(bingood, bingood.c.binfile == marvindb.dapdb.Junk.file_pk)

        # check for additional modifier criteria
        if self._parsed.functions:
            # loop over all functions
            for fxn in self._parsed.functions:
                # look up the function name in the marvinform dictionary
                try:
                    methodname = self.marvinform._param_fxn_lookup[fxn.fxnname]
                except KeyError as e:
                    self.reset()
                    raise MarvinError('Could not set function: {0}'.format(e))
                else:
                    # run the method
                    methodcall = self.__getattribute__(methodname)
                    methodcall(fxn)

    def _check_dapall_query(self):
        ''' Checks if the query is on the DAPall table.  '''

        isdapall = self._check_query('dapall')
        if isdapall:
            self.query = self._group_by()

    def _getGoodSpaxels(self):
        ''' Subquery - Counts the number of good spaxels

        Counts the number of good spaxels with binid != -1
        Uses the junk.bindid_pk != 9999 since this is known and set.
        Removes need to join to the binid table

        Returns:
            bincount (subquery):
                An SQLalchemy subquery to be joined into the main query object
        '''

        spaxelname = self._junkclass.__name__
        bincount = self.session.query(self._junkclass.file_pk.label('binfile'),
                                      func.count(self._junkclass.pk).label('goodcount'))

        # optionally add the filter if the table is SpaxelProp
        if 'CleanSpaxelProp' not in spaxelname:
            bincount = bincount.filter(self._junkclass.binid != -1)

        # group the results by file_pk
        bincount = bincount.group_by(self._junkclass.file_pk).subquery('bingood', with_labels=True)

        return bincount

    def _getCountOf(self, expression):
        ''' Subquery - Counts spaxels satisfying an expression

        Counts the number of spaxels of a given
        parameter above a certain value.

        Parameters:
            expression (str):
                The filter expression to parse

        Returns:
            valcount (subquery):
                An SQLalchemy subquery to be joined into the main query object

        Example:
            >>> expression = 'junk.emline_gflux_ha_6564 >= 25'
        '''

        # parse the expression into name, operator, value
        param, ops, value = self._parseExpression(expression)
        # look up the InstrumentedAttribute, Operator, and convert Value
        attribute = self.marvinform._param_form_lookup.mapToColumn(param)
        op = opdict[ops]
        value = float(value)
        # Build the subquery
        valcount = self.session.query(self._junkclass.file_pk.label('valfile'),
                                      (func.count(self._junkclass.pk)).label('valcount')).\
            filter(op(attribute, value)).\
            group_by(self._junkclass.file_pk).subquery('goodhacount', with_labels=True)

        return valcount

    def getPercent(self, fxn, **kwargs):
        ''' Query - Computes count comparisons

        Retrieves the number of objects that have satisfy a given expression
        in x% of good spaxels.  Expression is of the form
        Parameter Operand Value. This function is mapped to
        the "npergood" filter name.

        Syntax: fxnname(expression) operator value

        Parameters:
            fxn (str):
                The function condition used in the query filter

        Example:
            >>> fxn = 'npergood(junk.emline_gflux_ha_6564 > 25) >= 20'
            >>> Syntax: npergood() - function name
            >>>         npergood(expression) operator value
            >>>
            >>> Select objects that have Ha flux > 25 in more than
            >>> 20% of their (good) spaxels.
        '''

        # parse the function into name, condition, operator, and value
        name, condition, ops, value = self._parseFxn(fxn)
        percent = float(value) / 100.
        op = opdict[ops]

        # Retrieve the necessary subqueries
        bincount = self._getGoodSpaxels()
        valcount = self._getCountOf(condition)

        # Join to the main query
        self.query = self.query.join(bincount, bincount.c.binfile == self._junkclass.file_pk).\
            join(valcount, valcount.c.valfile == self._junkclass.file_pk).\
            filter(op(valcount.c.valcount, percent * bincount.c.goodcount))

        # Group the results by main defaultdatadb parameters,
        # so as not to include all spaxels
        newdefs = [d for d in self.defaultparams if 'spaxelprop' not in d]
        self.query = self._group_by(params=newdefs)
        # newdefaults = self.marvinform._param_form_lookup.mapToColumn(newdefs)
        # self.params = newdefs
        # self.query = self.query.from_self(*newdefaults).group_by(*newdefaults)

    def _parseFxn(self, fxn):
        ''' Parse a fxn condition '''
        return fxn.fxnname, fxn.fxncond, fxn.op, fxn.value

    def _parseExpression(self, expr):
        ''' Parse an expression '''
        return expr.fullname, expr.op, expr.value

    def _checkParsed(self):
        ''' Check the boolean parsed object

            check for function conditions vs normal.  This should be moved
            into SQLalchemy Boolean Search
        '''

        # Triggers for only one filter and it is a function condition
        if hasattr(self._parsed, 'fxn'):
            self._parsed.functions = [self._parsed]

        # Checks for shortcut names and replaces them in params
        # now redundant after pre-check on searchfilter
        for key, val in self._parsed.params.items():
            if key in self.marvinform._param_form_lookup._nameShortcuts.keys():
                newkey = self.marvinform._param_form_lookup._nameShortcuts[key]
                self._parsed.params.pop(key)
                self._parsed.params.update({newkey: val})
