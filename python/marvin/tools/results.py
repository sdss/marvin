#!/usr/bin/env python
# encoding: utf-8

# Licensed under a 3-clause BSD license.
#

from __future__ import print_function

import copy
import datetime
import json
import os
import warnings
from collections import namedtuple
from functools import wraps
from operator import add

import numpy as np
import six
from astropy.table import Table, hstack, vstack
from fuzzywuzzy import process
import marvin.utils.plot.scatter
from marvin import config, log
from marvin.api.api import Interaction
from marvin.core import marvin_pickle
from marvin.core.exceptions import (MarvinBreadCrumb, MarvinError, MarvinUserWarning)
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.modelcube import ModelCube
from marvin.tools.rss import RSS
from marvin.utils.datamodel.query import datamodel
from marvin.utils.datamodel.query.base import ParameterGroup
from marvin.utils.general import (downloadList, map_bins_to_column, temp_setattr,
                                  turn_off_ion)

try:
    import cPickle as pickle
except:
    import pickle

try:
    import pandas as pd
except ImportError:
    warnings.warn('Could not import pandas.', MarvinUserWarning)

__all__ = ['Results', 'ResultSet']

breadcrumb = MarvinBreadCrumb()


def local_mode_only(fxn):
    '''Decorator that bypasses function if in remote mode.'''

    @wraps(fxn)
    def wrapper(self, *args, **kwargs):
        if self.mode == 'remote':
            raise MarvinError('{0} not available in remote mode'.format(fxn.__name__))
        else:
            return fxn(self, *args, **kwargs)
    return wrapper


def remote_mode_only(fxn):
    '''Decorator that bypasses function if in local mode.'''

    @wraps(fxn)
    def wrapper(self, *args, **kwargs):
        if self.mode == 'local':
            raise MarvinError('{0} not available in local mode'.format(fxn.__name__))
        else:
            return fxn(self, *args, **kwargs)
    return wrapper


class ColumnGroup(ParameterGroup):
    ''' Subclass of Parameter Group '''

    def __repr__(self):
        ''' New repr for the Results Column Parameter Group '''
        old = list.__repr__(self)
        old = old.replace('>,', '>,\n')
        return ('<ParameterGroup name={0.name}, n_parameters={1}>\n '
                '{2}'.format(self, len(self), old))

    def __str__(self):
        ''' New string repr for prints '''
        return self.__repr__()


def marvintuple(name, params=None, **kwargs):
    ''' Custom namedtuple class factory for Marvin Results rows

    A class factory designed to create a new Marvin ResultRow class object. marvintuple
    creates a new class definition which can be instantiated.  Parameters can be pushed into
    an instance as individual arguments, or as key-value pairs.  See the
    `namedtuple <https://docs.python.org/2/library/collections.html#collections.namedtuple>`_
    for details on the Python container.

    Parameters:
        name (str):
            The name of the Class.  Required.
        params (str|list):
            The list of parameters to add as fields to the namedtuple.  Can be a list of names
            or a comma-separated string of names.

    Returns:
        a new namedtuple class

    Example:
        >>> # create new class with two fields
        >>> mt = marvintuple('Row', ['mangaid', 'plateifu'])
        >>>
        >>> # create a new instance of the class, with values
        >>> row = mt('1-209232', '8485-1901')
        >>> # or
        >>> row = mt(mangaid='1-209232', plateifu='8485-1901')

    '''

    # check the params input
    if params and isinstance(params, six.string_types):
        params = params.split(',') if ',' in params else [params]
        params = [p.strip() for p in params]

    # pop any extra keywords
    results = kwargs.pop('results', None)

    # create default namedtuple
    nt = namedtuple(name, params, **kwargs)

    def new_add(self, other):
        ''' Overloaded add to combine tuples without duplicates '''

        if self.release:
            assert self.release == other.release, 'Cannot add result rows from different releases'
        if self._search_filter:
            assert self._search_filter == other._search_filter, ('Cannot add result rows generated '
                                                                 'using different search filters')

        assert hasattr(self, 'plateifu') and hasattr(other, 'plateifu'), ("All rows must have a "
                                                                          "plateifu column to be able to add")

        assert self.plateifu == other.plateifu, 'The plateifus must be the same to add these rows'

        self_dict = self._asdict()
        other_dict = other._asdict()
        self_dict.update(other_dict)

        new_fields = tuple(self_dict.keys())
        marvin_row = marvintuple(self.__class__.__name__, new_fields, results=self._results)
        return marvin_row(**self_dict)

    # append new properties and overloaded methods
    nt.__add__ = new_add
    nt._results = results
    nt.release = results.release if results else None
    nt._search_filter = results.search_filter if results else None
    return nt


class ResultSet(list):
    ''' A Set of Results

    A list object representing a set of query results.  Each row of the list is a
    ResultRow object, which is a custom Marvin namedtuple object.  ResultSets can be
    extended column-wise or row-wise by adding them together.

    Parameters:
        _objects (list):
            A list of objects. Required.
        count (int):
            The count of objects in the current list
        totalcount (int):
            The total count of objects in the full results
        index (int):
            The index of the current set within the total set.
        columns (list):
            A list of columns accompanying this set
        results (Results):
            The Marvin Results object this set is a part of

    '''
    def __init__(self, _objects, **kwargs):
        list.__init__(self, _objects)
        self._results = kwargs.get('results', None)
        self.columns = kwargs.get('columns', None)
        self.count = kwargs.get('count', None)
        self.total = kwargs.get('total', None)
        self._populate_from_results()
        self.pages = int(np.ceil(self.total / float(self.count))) if self.count else 0
        self.index = kwargs.get('index') if kwargs.get('index') else 0
        self.end_index = self.index + self.count
        self.current_page = (int(self.index) + self.count) / self.count

    def __repr__(self):
        old = list.__repr__(self)
        return ('<ResultSet(set={0.current_page}/{0.pages}, index={0.index}:{0.end_index}, '
                'count_in_set={0.count}, total={0.total})>\n{1}'.format(self, old.replace('),', '),\n')))

    def __getitem__(self, value):
        if isinstance(value, six.string_types):
            value = str(value)
            if value in self.columns:
                colname = self.columns[value].remote
                rows = [row.__getattribute__(colname) for row in self]
            else:
                rows = [row for row in self if value in row]
            if rows:
                return rows[0] if len(rows) == 1 else rows
            else:
                raise ValueError('{0} not found in the list'.format(value))
        elif isinstance(value, int):
            return list.__getitem__(self, value)
        elif isinstance(value, slice):
            newset = list.__getitem__(self, value)
            return ResultSet(newset, index=int(value.start), count=len(newset), total=self.total, columns=self.columns, results=self._results)
        elif isinstance(value, np.ndarray):
            return np.array(self)[value]

    def __getslice__(self, start, stop):
        newset = list.__getslice__(self, start, stop)
        return ResultSet(newset, index=start, count=len(newset), total=self.total, columns=self.columns, results=self._results)

    def __add__(self, other):
        newresults = self._results

        if not isinstance(other, ResultSet):
            raise MarvinUserWarning('Can only add ResultSets together')

        # add elements
        if self.index == other.index:
            # column-wise add
            newcols = self.columns.full + [col.full for col in other.columns if col.full not in self.columns.full]
            parent = self._results.datamodel if self._results else None
            newcols = ColumnGroup('Columns', newcols, parent=parent)
            newresults.columns = newcols
            new_set = map(add, self, other)
        else:
            # row-wise add

            # warn if the subsets are not consecutive
            if abs(self.index - other.index) > self.count:
                warnings.warn('You are combining non-consectuive sets! '
                              'The indexing and ordering will be messed up')

            # copy the sets
            new_set = copy.copy(self) if self.index < other.index else copy.copy(other)
            set_b = copy.copy(other) if self.index < other.index else copy.copy(self)
            # filter out any rows that already exist in the set
            rows = [row for row in set_b if row not in new_set]
            # extend the set
            new_set.extend(rows)
            newcols = self.columns
            self.count = len(new_set)
            self.index = min(self.index, other.index)

        return ResultSet(new_set, count=self.count, total=self.total, index=self.index,
                         columns=newcols, results=newresults)

    def __radd__(self, other):
        return self.__add__(other)

    def _populate_from_results(self):
        ''' Populate some parameters from the results '''
        if self._results:
            self.columns = self._results.columns if not self.columns else self.columns
            self.choices = self.columns.list_params('remote')
            self.count = self._results.count if not self.count else self.count
            self.total = self._results.totalcount if not self.total else self.total
        else:
            self.count = self.count if self.count else len(self)
            self.total = self.total if self.total else len(self)

    def to_dict(self, name=None, format_type='listdict'):
        ''' Convert the ResultSet into a dictionary

        Converts the set of results into a list of dictionaries.  Optionally
        accepts a column name keyword to extract only that column.

        Parameters:
            name (str):
                Name of the column you wish to extract.  Default is None.
            format_type (str):
                The format of the output dictionary.  Can either be a list of dictionaries
                or a dictionary of lists.

        Returns:
            The output converted into dictionary format.
        '''

        keys = self.columns.list_params('remote')

        if format_type == 'listdict':
            if name:
                output = [{k: res.__getattribute__(k) for k in [name]} for res in self]
            else:
                output = [{k: res.__getattribute__(k) for k in keys} for res in self]
        elif format_type == 'dictlist':
            if name:
                output = {k: [res._asdict()[k] for res in self] for k in [name]}
            else:
                output = {k: [res._asdict()[k] for res in self] for k in keys}
        else:
            raise MarvinError('Cannot output dictionaries.  Check your input format_type.')
        return output

    def to_list(self):
        ''' Converts to a standard Python list object '''
        return list(self)

    def sort(self, name=None, reverse=False):
        ''' Sort the results

        In-place sorting of the result set.  This is the standard list sorting mechanism.
        When no name is specified, does standard list sorting with no key.

        Parameters:
            name (str):
                Column name to sort on.  Default is None.
            reverse (bool):
                If True, sorts in reverse (descending) order.

        Returns:
            A sorted list

        '''
        if name:
            colname = self.columns[name].remote
            return list.sort(self, key=lambda row: row.__getattribute__(colname), reverse=reverse)
        else:
            return list.sort(self)


class Results(object):
    ''' A class to handle results from queries on the MaNGA dataset

    Parameters:
        results (list):
            List of results satisfying the input Query
        query (object / str):
            The query used to produce these results. In local mode, the query is an
            SQLalchemy object that can be used to redo the query, or extract subsets
            of results from the query. In remote more, the query is a literal string
            representation of the SQL query.
        return_type (str):
            The MarvinTools object to convert the results into.  If initially set, the results
            are automaticaly converted into the specified Marvin Tool Object on initialization
        objects (list):
            The list of Marvin Tools objects created by returntype
        count (int):
            The number of objects in the returned query results
        totalcount (int):
            The total number of objects in the full query results
        mode ({'auto', 'local', 'remote'}):
            The load mode to use. See :doc:`Mode secision tree</mode_decision>`.
        chunk (int):
            For paginated results, the number of results to return.  Defaults to 10.
        start (int):
            For paginated results, the starting index value of the results.  Defaults to 0.
        end (int):
            For paginated results, the ending index value of the resutls.  Defaults to start+chunk.

    Attributes:
        count (int):  The count of objects in your current page of results
        totalcount (int): The total number of results in the query
        query_time (datetime): A datetime TimeDelta representation of the query runtime

    Returns:
        results: An object representing the Results entity

    Example:
        >>> f = 'nsa.z < 0.012 and ifu.name = 19*'
        >>> q = Query(search_filter=f)
        >>> r = q.run()
        >>> print(r)
        >>> Results(results=[(u'4-3602', u'1902', -9999.0), (u'4-3862', u'1902', -9999.0), (u'4-3293', u'1901', -9999.0), (u'4-3988', u'1901', -9999.0), (u'4-4602', u'1901', -9999.0)],
        >>>         query=<sqlalchemy.orm.query.Query object at 0x115217090>,
        >>>         count=64,
        >>>         mode=local)

    '''

    def __init__(self, results=None, mode=None, data_origin=None, release=None, count=None,
                 totalcount=None, runtime=None, response_time=None, chunk=None, start=None,
                 end=None, queryobj=None, query=None, search_filter=None, return_params=None,
                 return_type=None, limit=None, params=None, **kwargs):

        # basic parameters
        self.results = results
        self.mode = mode if mode else config.mode
        self.data_origin = data_origin
        self.objects = None

        # input query parameters
        self._queryobj = queryobj
        self._params = self._queryobj.params if self._queryobj else params
        self.release = self._queryobj.release if self._queryobj else release
        self.query = self._queryobj.query if self._queryobj else query
        self.return_type = self._queryobj.return_type if self._queryobj else return_type
        self.search_filter = self._queryobj.search_filter if self._queryobj else search_filter
        self.return_params = self._queryobj.return_params if self._queryobj else return_params
        self.limit = self._queryobj.limit if self._queryobj else limit

        # stat parameters
        self.datamodel = datamodel[self.release]
        self.count = count if count else len(self.results)
        self.totalcount = totalcount if totalcount else self.count
        self._runtime = runtime
        self.query_time = self._getRunTime() if self._runtime is not None else None
        self.response_time = response_time

        # ordering parameters
        self.chunk = chunk
        self.start = start
        self.end = end
        self.sortcol = None
        self.order = None

        # drop breadcrumb
        breadcrumb.drop(message='Initializing MarvinResults {0}'.format(self.__class__),
                        category=self.__class__)

        # Convert results to MarvinTuple
        if self.count > 0 and self.results:
            self._set_page()
            self._create_result_set(index=self.start)

        # Auto convert to Marvin Object
        if self.return_type:
            self.convertToTool(self.return_type)

    def __add__(self, other):
        assert isinstance(other, Results) is True, 'Can only add Marvin Results together'
        assert self.release == other.release, 'Cannot add Marvin Results from different releases'
        assert self.search_filter == other.search_filter, 'Cannot add Marvin Results with different search filters'
        results = self.results + other.results
        return_params = self.return_params + [p for p in other.return_params if p not in self.return_params]
        params = self._params + [p for p in other._params if p not in self._params]
        return Results(results=results, params=params, return_params=return_params, limit=self.limit,
                       search_filter=self.search_filter, count=len(results), totalcount=self.totalcount,
                       release=self.release, mode=self.mode)

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return ('Marvin Results(query={0}, totalcount={1}, count={2}, mode={3})'.format(self.search_filter, self.totalcount, self.count, self.mode))

    def showQuery(self):
        ''' Displays the literal SQL query used to generate the Results objects

        Returns:
            querystring (str):
                A string representation of the SQL query
        '''

        # check unicode or str
        isstr = isinstance(self.query, six.string_types)

        # return the string query or compile the real query
        if isstr:
            return self.query
        else:
            return str(self.query.statement.compile(compile_kwargs={'literal_binds': True}))

    def _getRunTime(self):
        ''' Sets the query runtime as a datetime timedelta object '''
        if isinstance(self._runtime, dict):
            return datetime.timedelta(**self._runtime)
        else:
            return self._runtime

    def download(self, images=False, limit=None):
        ''' Download results via sdss_access

        Uses sdss_access to download the query results via rsync.
        Downloads them to the local sas. The data type downloaded
        is indicated by the returntype parameter

        i.e. $SAS_BASE_DIR/mangawork/manga/spectro/redux/...

        Parameters:
            images (bool):
                Set to only download the images of the query results
            limit (int):
                A limit of the number of results to download

        Returns:
            NA: na

        Example:
            >>> r = q.run()
            >>> r.returntype = 'cube'
            >>> r.download()
        '''

        plateifu = self.getListOf('plateifu')
        if images:
            tmp = get_images_by_list(plateifu, releas=self.release, download=True)
        else:
            downloadList(plateifu, dltype=self.return_type, limit=limit)

    def sort(self, name, order='asc'):
        ''' Sort the set of results by column name

        Sorts the results (in place) by a given parameter / column name.  Sets
        the results to the new sorted results.

        Parameters:
            name (str):
                The column name to sort on
            order ({'asc', 'desc'}):
                To sort in ascending or descending order.  Default is asc.

        Example:
            >>> r = q.run()
            >>> r.getColumns()
            >>> [u'mangaid', u'name', u'nsa.z']
            >>> r.results
            >>> [(u'4-3988', u'1901', -9999.0),
            >>>  (u'4-3862', u'1902', -9999.0),
            >>>  (u'4-3293', u'1901', -9999.0),
            >>>  (u'4-3602', u'1902', -9999.0),
            >>>  (u'4-4602', u'1901', -9999.0)]

            >>> # Sort the results by mangaid
            >>> r.sort('mangaid')
            >>> [(u'4-3293', u'1901', -9999.0),
            >>>  (u'4-3602', u'1902', -9999.0),
            >>>  (u'4-3862', u'1902', -9999.0),
            >>>  (u'4-3988', u'1901', -9999.0),
            >>>  (u'4-4602', u'1901', -9999.0)]

            >>> # Sort the results by IFU name in descending order
            >>> r.sort('ifu.name', order='desc')
            >>> [(u'4-3602', u'1902', -9999.0),
            >>>  (u'4-3862', u'1902', -9999.0),
            >>>  (u'4-3293', u'1901', -9999.0),
            >>>  (u'4-3988', u'1901', -9999.0),
            >>>  (u'4-4602', u'1901', -9999.0)]
        '''
        remotename = self._check_column(name, 'remote')
        self.sortcol = remotename
        self.order = order

        if self.mode == 'local':
            reverse = True if order == 'desc' else False
            self.getAll()
            self.results.sort(remotename, reverse=reverse)
            self.results = self.results[0:self.limit]
        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['querycubes']['url']
            params = {'searchfilter': self.search_filter, 'returnparams': self.return_params,
                      'sort': remotename, 'order': order, 'limit': self.limit}

            self._interaction(url, params, create_set=True, calltype='Sort')

        return self.results

    def toTable(self):
        ''' Output the results as an Astropy Table

        Uses the Python Astropy package

        Parameters:
            None

        Returns:
            tableres:
                Astropy Table

        Example:
            >>> r = q.run()
            >>> r.toTable()
            >>> <Table length=5>
            >>> mangaid    name   nsa.z
            >>> unicode6 unicode4   float64
            >>> -------- -------- ------------
            >>>   4-3602     1902      -9999.0
            >>>   4-3862     1902      -9999.0
            >>>   4-3293     1901      -9999.0
            >>>   4-3988     1901      -9999.0
            >>>   4-4602     1901      -9999.0
        '''
        try:
            tabres = Table(rows=self.results, names=self.columns.full)
        except ValueError as e:
            raise MarvinError('Could not make astropy Table from results: {0}'.format(e))
        return tabres

    def merge_tables(self, tables, direction='vert', **kwargs):
        ''' Merges a list of Astropy tables of results together

        Combines two Astropy tables using either the Astropy
        vstack or hstack method.  vstack refers to vertical stacking of table rows.
        hstack refers to horizonal stacking of table columns.  hstack assumes the rows in each
        table refer to the same object.  Buyer beware: stacking tables without proper understanding
        of your rows and columns may results in deleterious results.

        merge_tables also accepts all keyword arguments that Astropy vstack and hstack method do.
        See `vstack <http://docs.astropy.org/en/stable/table/operations.html#stack-vertically>`_
        See `hstack <http://docs.astropy.org/en/stable/table/operations.html#stack-horizontally>`_

        Parameters:
            tables (list):
                A list of Astropy Table objects.  Required.
            direction (str):
                The direction of the table stacking, either vertical ('vert') or horizontal ('hor').
                Default is 'vert'.  Direction string can be fuzzy.

        Returns:
            A new Astropy table that is the stacked combination of all input tables

        Example:
            >>> # query 1
            >>> q, r = doQuery(search_filter='nsa.z < 0.1', returnparams=['g_r', 'cube.ra', 'cube.dec'])
            >>> # query 2
            >>> q2, r2 = doQuery(search_filter='nsa.z < 0.1')
            >>>
            >>> # convert to tables
            >>> table_1 = r.toTable()
            >>> table_2 = r2.toTable()
            >>> tables = [table_1, table_2]
            >>>
            >>> # vertical (row) stacking
            >>> r.merge_tables(tables, direction='vert')
            >>> # horizontal (column) stacking
            >>> r.merge_tables(tables, direction='hor')

        '''
        choices = ['vertical', 'horizontal']
        stackdir, score = process.extractOne(direction, choices)
        if stackdir == 'vertical':
            return vstack(tables, **kwargs)
        elif stackdir == 'horizontal':
            return hstack(tables, **kwargs)

    def toFits(self, filename='myresults.fits', overwrite=False):
        ''' Output the results as a FITS file

        Writes a new FITS file from search results using
        the astropy Table.write()

        Parameters:
            filename (str):
                Name of FITS file to output
            overwrite (bool):
                Set to True to overwrite an existing file

        '''
        myext = os.path.splitext(filename)[1]
        if not myext:
            filename = filename + '.fits'
        table = self.toTable()
        table.write(filename, format='fits', overwrite=overwrite)
        print('Writing new FITS file {0}'.format(filename))

    def toCSV(self, filename='myresults.csv', overwrite=False):
        ''' Output the results as a CSV file

        Writes a new CSV file from search results using
        the astropy Table.write()

        Parameters:
            filename (str):
                Name of CSV file to output
            overwrite (bool):
                Set to True to overwrite an existing file

        '''
        myext = os.path.splitext(filename)[1]
        if not myext:
            filename = filename + '.csv'
        table = self.toTable()
        table.write(filename, format='csv', overwrite=overwrite)
        print('Writing new CSV file {0}'.format(filename))

    def toDF(self):
        '''Call toDataFrame().
        '''
        return self.toDataFrame()

    def toDataFrame(self):
        '''Output the results as an pandas dataframe.

        Uses the pandas package.

        Parameters:
            None

        Returns:
            dfres:
                pandas dataframe

        Example:
            >>> r = q.run()
            >>> r.toDataFrame()
            mangaid  plate   name     nsa_mstar         z
            0  1-22286   7992  12704  1.702470e+11  0.099954
            1  1-22301   7992   6101  9.369260e+10  0.105153
            2  1-22414   7992   6103  7.489660e+10  0.092272
            3  1-22942   7992  12705  8.470360e+10  0.104958
            4  1-22948   7992   9102  1.023530e+11  0.119399
        '''
        try:
            dfres = pd.DataFrame(self.results.to_list())
        except (ValueError, NameError) as e:
            raise MarvinError('Could not make pandas dataframe from results: {0}'.format(e))
        return dfres

    def _create_result_set(self, index=None, rows=None):
        ''' Creates a Marvin ResultSet

        Parameters:
            index (int):
                The starting index of the result subset
            rows (list|ResultSet):
                A list of rows containing the value data to input into the ResultSet

        Returns:
            creates a marvin ResultSet and sets it as the results attribute

        '''

        # grab the columns from the results
        self.columns = self.getColumns()
        ntnames = self.columns.list_params('remote')
        # dynamically create a new ResultRow Class
        rows = rows if rows else self.results
        row_is_dict = isinstance(rows[0], dict)
        if not isinstance(rows, ResultSet):
            nt = marvintuple('ResultRow', ntnames, results=self)
            if row_is_dict:
                results = [nt(**r) for r in rows]
            else:
                results = [nt(*r) for r in rows]
        else:
            results = rows
        self.count = len(results)
        # Build the ResultSet
        self.results = ResultSet(results, count=self.count, total=self.totalcount, index=index, results=self)

    def _set_page(self):
        ''' Set the page of the data '''
        if self.start and self.end:
            self.chunk = (self.end - self.start)
        else:
            self.chunk = self.chunk if self.chunk else self.limit if self.limit else 100
            self.start = 0
            self.end = self.start + self.chunk

        self.pages = int(np.ceil(self.totalcount / float(self.count))) if self.count else 0
        self.index = self.start
        self.current_page = (int(self.index) + self.count) / self.count

    def save(self, path=None, overwrite=False):
        ''' Save the results as a pickle object

        Parameters:
            path (str):
                Filepath and name of the pickled object
            overwrite (bool):
                Set this to overwrite an existing pickled file

        Returns:
            path (str):
                The filepath and name of the pickled object

        '''

        # set the filename and path
        sf = self.search_filter.replace(' ', '') if self.search_filter else 'anon'
        # set the path
        if not path:
            path = os.path.expanduser('~/marvin_results_{0}.mpf'.format(sf))

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

        # convert results into a dict
        dict_results = self.results.to_dict()

        # set bad pickled attributes to None
        attrs = ['results', 'datamodel', 'columns', '_queryobj']
        vals = [dict_results, None, None, None]
        isnotstr = not isinstance(self.query, six.string_types)
        if isnotstr:
            attrs += ['query']
            vals += [None]

        # pickle the results
        try:
            with temp_setattr(self, attrs, vals):
                pickle.dump(self, open(path, 'wb'), protocol=-1)
        except Exception as ee:
            if os.path.exists(path):
                os.remove(path)
            raise MarvinError('Error found while pickling: {0}'.format(str(ee)))

        return path

    @classmethod
    def restore(cls, path, delete=False):
        ''' Restore a pickled Results object

        Parameters:
            path (str):
                The filename and path to the pickled object

            delete (bool):
                Turn this on to delete the pickled fil upon restore

        Returns:
            Results (instance):
                The instantiated Marvin Results class
        '''
        obj = marvin_pickle.restore(path, delete=delete)
        obj.datamodel = datamodel[obj.release]
        obj._create_result_set()
        obj.getColumns()
        return obj

    def toJson(self):
        ''' Output the results as a JSON object

        Uses Python json package to convert the results to JSON representation

        Parameters:
            None

        Returns:
            jsonres:
                JSONed results

        Example:
            >>> r = q.run()
            >>> r.toJson()
            >>> '[["4-3602", "1902", -9999.0], ["4-3862", "1902", -9999.0], ["4-3293", "1901", -9999.0],
            >>>   ["4-3988", "1901", -9999.0], ["4-4602", "1901", -9999.0]]'
        '''
        try:
            jsonres = json.dumps(self.results)
        except TypeError as e:
            raise MarvinError('Results not JSON-ifiable. Check the format of results: {0}'.format(e))
        return jsonres

    def getColumns(self):
        ''' Get the columns of the returned reults

        Returns a ParameterGroup containing the columns from the
        returned results.  Each row of the ParameterGroup is a
        QueryParameter.

        Returns:
            columns (list):
                A list of column names from the results

        Example:
            >>> r = q.run()
            >>> cols = r.getColumns()
            >>> print(cols)
            >>> [u'mangaid', u'name', u'nsa.z']

        '''
        try:
            self.columns = ColumnGroup('Columns', self._params, parent=self.datamodel)
        except Exception as e:
            raise MarvinError('Could not create query columns: {0}'.format(e))
        return self.columns

    def _interaction(self, url, params, calltype='', create_set=None, **kwargs):
        ''' Perform a remote Interaction call

        Parameters:
            url (str):
                The url of the request
            params (dict):
                A dictionary of parameters (get or post) to send with the request
            calltype (str):
                The method call sending the request
            create_set (bool):
                If True, sets the response output as the new results and creates a
                new named tuple set

        Returns:
            output:
                The output data from the request

        Raises:
            MarvinError: Raises on any HTTP Request error

        '''

        # check if the returnparams parameter is in the proper format
        if 'returnparams' in params:
            return_params = params.get('returnparams', None)
            if return_params and isinstance(return_params, list):
                params['returnparams'] = ','.join(return_params)

        # check if we're getting all results
        datastream = calltype == 'getAll'

        # send the request
        try:
            ii = Interaction(route=url, params=params, stream=True, datastream=datastream)
        except MarvinError as e:
            raise MarvinError('API Query {0} call failed: {1}'.format(calltype, e))
        else:
            remotes = self._queryobj._get_remote_parameters(ii)
            output = remotes['results']
            self.response_time = remotes['response_time']
            self._runtime = remotes['runtime']
            self.query_time = self._getRunTime()
            index = kwargs.get('index', None)
            if create_set:
                self._create_result_set(index=index, rows=output)
            else:
                return output

    def _check_column(self, name, name_type):
        ''' Check if a name exists as a column '''
        try:
            name_in_col = name in self.columns
        except KeyError as e:
            raise MarvinError('Column {0} not found in results: {1}'.format(name, e))
        else:
            assert name_type in ['full', 'remote', 'name', 'short', 'display'], \
                'name_type must be one of "full, remote, name, short, display"'
            return self.columns[str(name)].__getattribute__(name_type)

    def getListOf(self, name=None, to_json=False, to_ndarray=False, return_all=None):
        ''' Extract a list of a single parameter from results

        Parameters:
            name (str):
                Name of the parameter name to return.  If not specified,
                it returns all parameters.
            to_json (bool):
                True/False boolean to convert the output into a JSON format
            to_ndarray (bool):
                True/False boolean to convert the output into a Numpy array
            return_all (bool):
                if True, returns the entire result set for that column

        Returns:
            output (list):
                A list of results for one parameter

        Example:
            >>> r = q.run()
            >>> r.getListOf('mangaid')
            >>> [u'4-3988', u'4-3862', u'4-3293', u'4-3602', u'4-4602']

        Raises:
            AssertionError:
                Raised when no name is specified.
        '''

        assert name, 'Must specify a column name'

        # check column name and get full name
        fullname = self._check_column(name, 'full')

        # deal with the output
        if return_all:
            # # grab all of that column
            url = config.urlmap['api']['getcolumn']['url'].format(colname=fullname)
            params = {'searchfilter': self.search_filter, 'format_type': 'list',
                      'return_all': True, 'returnparams': self.return_params}
            output = self._interaction(url, params, calltype='getList')
        else:
            # only deal with current page
            output = self.results[name] if self.results.count > 1 else [self.results[name]]

        if to_json:
            output = json.dumps(output) if output else None

        if to_ndarray:
            output = np.array(output) if output else None

        return output

    def getDictOf(self, name=None, format_type='listdict', to_json=False, return_all=None):
        ''' Get a dictionary of specified parameters

        Parameters:
            name (str):
                Name of the parameter name to return.  If not specified,
                it returns all parameters.
            format_type ({'listdict', 'dictlist'}):
                The format of the results. Listdict is a list of dictionaries.
                Dictlist is a dictionary of lists. Default is listdict.
            to_json (bool):
                True/False boolean to convert the output into a JSON format
            return_all (bool):
                if True, returns the entire result set for that column

        Returns:
            output (list, dict):
                Can be either a list of dictionaries, or a dictionary of lists

        Example:
            >>> # get some results
            >>> r = q.run()
            >>> # Get a list of dictionaries
            >>> r.getDictOf(format_type='listdict')
            >>> [{'cube.mangaid': u'4-3988', 'ifu.name': u'1901', 'nsa.z': -9999.0},
            >>>  {'cube.mangaid': u'4-3862', 'ifu.name': u'1902', 'nsa.z': -9999.0},
            >>>  {'cube.mangaid': u'4-3293', 'ifu.name': u'1901', 'nsa.z': -9999.0},
            >>>  {'cube.mangaid': u'4-3602', 'ifu.name': u'1902', 'nsa.z': -9999.0},
            >>>  {'cube.mangaid': u'4-4602', 'ifu.name': u'1901', 'nsa.z': -9999.0}]

            >>> # Get a dictionary of lists
            >>> r.getDictOf(format_type='dictlist')
            >>> {'cube.mangaid': [u'4-3988', u'4-3862', u'4-3293', u'4-3602', u'4-4602'],
            >>>  'ifu.name': [u'1901', u'1902', u'1901', u'1902', u'1901'],
            >>>  'nsa.z': [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0]}

            >>> # Get a dictionary of only one parameter
            >>> r.getDictOf('mangaid')
            >>> [{'cube.mangaid': u'4-3988'},
            >>>  {'cube.mangaid': u'4-3862'},
            >>>  {'cube.mangaid': u'4-3293'},
            >>>  {'cube.mangaid': u'4-3602'},
            >>>  {'cube.mangaid': u'4-4602'}]

        '''

        # get the remote and full name
        remotename = self._check_column(name, 'remote') if name else None
        fullname = self._check_column(name, 'full') if name else None

        # create the dictionary
        output = self.results.to_dict(name=remotename, format_type=format_type)

        # deal with the output
        if return_all:
            # grab all or of a specific column
            params = {'searchfilter': self.search_filter, 'return_all': True,
                      'format_type': format_type, 'returnparams': self.return_params}
            url = config.urlmap['api']['getcolumn']['url'].format(colname=fullname)
            output = self._interaction(url, params, calltype='getDict')
        else:
            # only deal with current page
            output = self.results.to_dict(name=remotename, format_type=format_type)

        if to_json:
            output = json.dumps(output) if output else None

        return output

    def loop(self, chunk=None):
        ''' Loop over the full set of results

        Starts a loop to collect all the results (in chunks)
        until the current count reaches the total number
        of results.  Uses extendSet.

        Parameters:
            chunk (int):
                The number of objects to return

        Example:
            >>> # get some results from a query
            >>> r = q.run()
            >>> # start a loop, grabbing in chunks of 400
            >>> r.loop(chunk=400)

        '''

        while self.count < self.totalcount:
            self.extendSet(chunk=chunk)

    def extendSet(self, chunk=None, start=None):
        ''' Extend the Result set with the next page

        Extends the current ResultSet with the next page of results
        or a specified page.  Calls either getNext or getSubset.

        Parameters:
            chunk (int):
                The number of objects to return
            start (int):
                The starting index of your subset extraction

        Returns:
            A new results set

        Example:
            >>> # run a query
            >>> r = q.run()
            >>> # extend the current result set with the next page
            >>> r.extendSet()
            >>>

        See Also:
            getNext, getSubset

        '''

        oldset = copy.copy(self.results)
        if start is not None:
            nextset = self.getSubset(start, limit=chunk)
        else:
            nextset = self.getNext(chunk=chunk)
        newset = oldset + nextset
        self.count = len(newset)
        self.results = newset

    def getNext(self, chunk=None):
        ''' Retrieve the next chunk of results

        Returns the next chunk of results from the query.
        from start to end in units of chunk.  Used with getPrevious
        to paginate through a long list of results

        Parameters:
            chunk (int):
                The number of objects to return

        Returns:
            results (list):
                A list of query results

        Example:
            >>> r = q.run()
            >>> r.getNext(5)
            >>> Retrieving next 5, from 35 to 40
            >>> [(u'4-4231', u'1902', -9999.0),
            >>>  (u'4-14340', u'1901', -9999.0),
            >>>  (u'4-14510', u'1902', -9999.0),
            >>>  (u'4-13634', u'1901', -9999.0),
            >>>  (u'4-13538', u'1902', -9999.0)]

        See Also:
            getAll, getPrevious, getSubset

        '''

        if chunk and chunk < 0:
            warnings.warn('Chunk cannot be negative. Setting to {0}'.format(self.chunk), MarvinUserWarning)
            chunk = self.chunk

        newstart = self.end
        self.chunk = chunk if chunk else self.chunk
        newend = newstart + self.chunk

        # This handles cases when the number of results is < total
        if self.totalcount == self.count:
            warnings.warn('You have all the results.  Cannot go forward', MarvinUserWarning)
            return self.results

        # This handles the end edge case
        if newend > self.totalcount:
            warnings.warn('You have reached the end.', MarvinUserWarning)
            newend = self.totalcount
            newstart = self.end

        # This grabs the next chunk
        log.info('Retrieving next {0}, from {1} to {2}'.format(self.chunk, newstart, newend))
        if self.mode == 'local':
            self.results = self.query.slice(newstart, newend).all()
            if self.results:
                self._create_result_set(index=newstart)
        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['getsubset']['url']
            params = {'searchfilter': self.search_filter, 'returnparams': self.return_params,
                      'start': newstart, 'end': newend, 'limit': chunk,
                      'sort': self.sortcol, 'order': self.order}
            self._interaction(url, params, calltype='getNext', create_set=True,
                              index=newstart)

        self.start = newstart
        self.end = newend
        self.count = len(self.results)

        if self.return_type:
            self.convertToTool(self.return_type)

        return self.results

    def getPrevious(self, chunk=None):
        ''' Retrieve the previous chunk of results.

        Returns a previous chunk of results from the query.
        from start to end in units of chunk.  Used with getNext
        to paginate through a long list of results

        Parameters:
            chunk (int):
                The number of objects to return

        Returns:
            results (list):
                A list of query results

        Example:
            >>> r = q.run()
            >>> r.getPrevious(5)
            >>> Retrieving previous 5, from 30 to 35
            >>> [(u'4-3988', u'1901', -9999.0),
            >>>  (u'4-3862', u'1902', -9999.0),
            >>>  (u'4-3293', u'1901', -9999.0),
            >>>  (u'4-3602', u'1902', -9999.0),
            >>>  (u'4-4602', u'1901', -9999.0)]

        See Also:
            getNext, getAll, getSubset

         '''

        if chunk and chunk < 0:
            warnings.warn('Chunk cannot be negative. Setting to {0}'.format(self.chunk), MarvinUserWarning)
            chunk = self.chunk

        newend = self.start
        self.chunk = chunk if chunk else self.chunk
        newstart = newend - self.chunk

        # This handles cases when the number of results is < total
        if self.totalcount == self.count:
            warnings.warn('You have all the results.  Cannot go back', MarvinUserWarning)
            return self.results

        # This handles the start edge case
        if newstart < 0:
            warnings.warn('You have reached the beginning.', MarvinUserWarning)
            newstart = 0
            newend = self.start

        # This grabs the previous chunk
        log.info('Retrieving previous {0}, from {1} to {2}'.format(self.chunk, newstart, newend))
        if self.mode == 'local':
            self.results = self.query.slice(newstart, newend).all()
            if self.results:
                self._create_result_set(index=newstart)
        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['getsubset']['url']

            params = {'searchfilter': self.search_filter, 'returnparams': self.return_params,
                      'start': newstart, 'end': newend, 'limit': chunk,
                      'sort': self.sortcol, 'order': self.order}
            self._interaction(url, params, calltype='getPrevious', create_set=True,
                              index=newstart)

        self.start = newstart
        self.end = newend
        self.count = len(self.results)

        if self.return_type:
            self.convertToTool(self.return_type)

        return self.results

    def getSubset(self, start, limit=None):
        ''' Extracts a subset of results

        Parameters:
            start (int):
                The starting index of your subset extraction
            limit (int):
                The limiting number of results to return.

        Returns:
            results (list):
                A list of query results

        Example:
            >>> r = q.run()
            >>> r.getSubset(0, 10)
            >>> [(u'14-12', u'1901', -9999.0),
            >>> (u'14-13', u'1902', -9999.0),
            >>> (u'27-134', u'1901', -9999.0),
            >>> (u'27-100', u'1902', -9999.0),
            >>> (u'27-762', u'1901', -9999.0),
            >>> (u'27-759', u'1902', -9999.0),
            >>> (u'27-827', u'1901', -9999.0),
            >>> (u'27-828', u'1902', -9999.0),
            >>> (u'27-1170', u'1901', -9999.0),
            >>> (u'27-1167', u'1902', -9999.0)]

        See Also:
            getNext, getPrevious, getAll

        '''

        if not limit:
            limit = self.chunk

        if limit < 0:
            warnings.warn('Limit cannot be negative. Setting to {0}'.format(self.chunk), MarvinUserWarning)
            limit = self.chunk

        start = 0 if int(start) < 0 else int(start)
        end = start + int(limit)
        self.start = start
        self.end = end
        self.chunk = limit
        if self.mode == 'local':
            self.results = self.query.slice(start, end).all()
            if self.results:
                self._create_result_set(index=start)
        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['getsubset']['url']

            params = {'searchfilter': self.search_filter, 'returnparams': self.return_params,
                      'start': start, 'end': end, 'limit': limit,
                      'sort': self.sortcol, 'order': self.order}
            self._interaction(url, params, calltype='getSubset', create_set=True, index=start)

        self.count = len(self.results)
        if self.return_type:
            self.convertToTool(self.return_type)

        return self.results

    def getAll(self, force=False):
        ''' Retrieve all of the results of a query

        Attempts to return all the results of a query.  The efficiency of this
        method depends heavily on how many rows and columns you wish to return.

        A cutoff limit is applied for results with more than 500,000 rows or
        results with more than 25 columns.

        Parameters
            force (bool):
                If True, force attempt to download everything

        Returns:
            The full list of query results.

        See Also:
            getNext, getPrevious, getSubset, loop

        '''

        if (self.totalcount > 500000 or len(self.columns) > 25) and not force:
            raise MarvinUserWarning("Cannot retrieve all results. The total number of requested "
                                    "rows or columns is too high. Please use the getNext, getPrevious, "
                                    "getSubset, or loop methods to retrieve pages.")

        if self.mode == 'local':
            self.results = self.query.from_self().all()
            self._create_result_set()
        elif self.mode == 'remote':
            # Get the query route
            url = config.urlmap['api']['querycubes']['url']

            params = {'searchfilter': self.search_filter, 'return_all': True,
                      'returnparams': self.return_params, 'limit': self.limit,
                      'sort': self.sortcol, 'order': self.order}
            self._interaction(url, params, calltype='getAll', create_set=True)
            self.count = self.totalcount
            print('Returned all {0} results'.format(self.totalcount))

    def convertToTool(self, tooltype, mode='auto', limit=None):
        ''' Converts the list of results into Marvin Tool objects

        Creates a list of Marvin Tool objects from a set of query results.
        The new list is stored in the Results.objects property.
        If the Query.returntype parameter is specified, then the Results object
        will automatically convert the results to the desired Tool on initialization.

        Parameters:
            tooltype (str):
                The requested Marvin Tool object that the results are converted into.
                Overrides the returntype parameter.  If not set, defaults
                to the returntype parameter.
            limit (int):
                Limit the number of results you convert to Marvin tools.  Useful
                for extremely large result sets.  Default is None.
            mode (str):
                The mode to use when attempting to convert to Tool. Default mode
                is to use the mode internal to Results. (most often remote mode)

        Example:
            >>> # Get the results from some query
            >>> r = q.run()
            >>> r.results
            >>> [NamedTuple(mangaid=u'14-12', name=u'1901', nsa.z=-9999.0),
            >>>  NamedTuple(mangaid=u'14-13', name=u'1902', nsa.z=-9999.0),
            >>>  NamedTuple(mangaid=u'27-134', name=u'1901', nsa.z=-9999.0),
            >>>  NamedTuple(mangaid=u'27-100', name=u'1902', nsa.z=-9999.0),
            >>>  NamedTuple(mangaid=u'27-762', name=u'1901', nsa.z=-9999.0)]

            >>> # convert results to Marvin Cube tools
            >>> r.convertToTool('cube')
            >>> r.objects
            >>> [<Marvin Cube (plateifu='7444-1901', mode='remote', data_origin='api')>,
            >>>  <Marvin Cube (plateifu='7444-1902', mode='remote', data_origin='api')>,
            >>>  <Marvin Cube (plateifu='7995-1901', mode='remote', data_origin='api')>,
            >>>  <Marvin Cube (plateifu='7995-1902', mode='remote', data_origin='api')>,
            >>>  <Marvin Cube (plateifu='8000-1901', mode='remote', data_origin='api')>]

        '''

        # set the desired tool type
        toollist = ['cube', 'spaxel', 'maps', 'rss', 'modelcube']
        tooltype = tooltype if tooltype else self.return_type
        assert tooltype in toollist, 'Returned tool type must be one of {0}'.format(toollist)

        # get the parameter list to check against
        paramlist = self.columns.full

        print('Converting results to Marvin {0} objects'.format(tooltype.title()))
        if tooltype == 'cube':
            self.objects = [self._get_object(Cube, plateifu=res.plateifu, mode=mode) for res in self.results[0:limit]]
        elif tooltype == 'maps':

            isbin = 'bintype.name' in paramlist
            istemp = 'template.name' in paramlist
            self.objects = []

            for res in self.results[0:limit]:
                mapkwargs = {'mode': mode, 'plateifu': res.plateifu}

                if isbin:
                    binval = res.bintype_name
                    mapkwargs['bintype'] = binval

                if istemp:
                    tempval = res.template_name
                    mapkwargs['template_kin'] = tempval

                self.objects.append(self._get_object(Maps, **mapkwargs))
        elif tooltype == 'spaxel':

            assert 'spaxelprop.x' in paramlist and 'spaxelprop.y' in paramlist, \
                   'Parameters must include spaxelprop.x and y in order to convert to Marvin Spaxel.'

            self.objects = []

            tab = self.toTable()
            uniq_plateifus = list(set(self.getListOf('plateifu')))

            for plateifu in uniq_plateifus:
                c = self._get_object(Cube, plateifu=plateifu, mode=mode)
                univals = tab['cube.plateifu'] == plateifu
                x = tab[univals]['spaxelprop.x'].tolist()
                y = tab[univals]['spaxelprop.y'].tolist()
                spaxels = c[y, x]
                self.objects.extend(spaxels)
        elif tooltype == 'rss':
            self.objects = [self._get_object(RSS, plateifu=res.plateifu, mode=mode) for res in self.results[0:limit]]
        elif tooltype == 'modelcube':

            isbin = 'bintype.name' in paramlist
            istemp = 'template.name' in paramlist
            self.objects = []

            assert self.release != 'MPL-4', "ModelCubes require a release of MPL-5 and up"

            for res in self.results[0:limit]:
                mapkwargs = {'mode': mode, 'plateifu': res.plateifu}

                if isbin:
                    binval = res.bintype_name
                    mapkwargs['bintype'] = binval

                if istemp:
                    tempval = res.template_name
                    mapkwargs['template_kin'] = tempval

                self.objects.append(self._get_object(ModelCube, **mapkwargs))

    @staticmethod
    def _get_object(obj, **kwargs):
        ''' Return a Marvin object or an error message

        To preserve the lengths of self.results and self.objects, it will
        either return an instance or an error message in its place

        Parameters:
            obj (object):
                A Marvin Class object
            kwargs:
                Any set of parameters to instantiate a Marvin object

        Returns:
            The Marvin instance or an error message if it failed
        '''

        try:
            inst = obj(**kwargs)
        except MarvinError as e:
            plateifu = kwargs.get('plateifu', '')
            inst = 'Error creating {0} for {1}: {2}'.format(obj.__name__, plateifu, e)

        return inst

    def plot(self, x_name, y_name, **kwargs):
        ''' Make a scatter plot from two columns of results

        Creates a Matplotlib scatter plot from Results columns.
        Accepts as input two string column names.  Will extract the total
        entire column (if not already available) and plot them.  Creates
        a scatter plot with (optionally) adjoining 1-d histograms for each column.

        See :meth:`marvin.utils.plot.scatter.plot` and
        :meth:`marvin.utils.plot.scatter.hist` for details.

        Parameters:
            x_name (str):
                The name of the x-column of data. Required
            y_name (str):
                The name of the y-column of data. Required
            return_plateifus (bool):
                If True, includes the plateifus in each histogram bin in the
                histogram output.  Default is True.
            return_figure (bool):
                Set to False to not return the Figure and Axis object. Defaults to True.
            show_plot (bool):
                Set to False to not show the interactive plot
            **kwargs (dict):
                Any other keyword argument that will be passed to Marvin's
                scatter and hist plotting methods

        Returns:
            The figure, axes, and histogram data from the plotting function

        Example:
            >>> # do a query and get the results
            >>> q = Query(search_filter='nsa.z < 0.1', returnparams=['nsa.elpetro_ba', 'g_r'])
            >>> r = q.run()
            >>> # plot the total columns of Redshift vs g-r magnitude
            >>> fig, axes, hist_data = r.plot('nsa.z', 'g_r')

        '''

        assert all([x_name, y_name]), 'Must provide both an x and y column'
        return_plateifus = kwargs.pop('return_plateifus', True)
        with_hist = kwargs.get('with_hist', True)
        show_plot = kwargs.pop('show_plot', True)
        return_figure = kwargs.get('return_figure', True)

        # get the named column
        x_col = self.columns[x_name]
        y_col = self.columns[y_name]

        # get the values of the two columns
        if self.count != self.totalcount:
            x_data = self.getListOf(x_name, return_all=True)
            y_data = self.getListOf(y_name, return_all=True)
        else:
            x_data = self.results[x_name]
            y_data = self.results[y_name]

        with turn_off_ion(show_plot=show_plot):
            output = marvin.utils.plot.scatter.plot(x_data, y_data, xlabel=x_col, ylabel=y_col, **kwargs)

        # computes a list of plateifus in each bin
        if return_plateifus and with_hist:
            plateifus = self.getListOf('plateifu', return_all=True)
            hdata = output[2] if return_figure else output
            if 'xhist' in hdata:
                hdata['xhist']['bins_plateifu'] = map_bins_to_column(plateifus, hdata['xhist']['indices'])
            if 'yhist' in hdata:
                hdata['yhist']['bins_plateifu'] = map_bins_to_column(plateifus, hdata['yhist']['indices'])
            output = output[0:2] + (hdata,) if return_figure else hdata

        return output

    def hist(self, name, **kwargs):
        ''' Make a histogram for a given column of the results

        Creates a Matplotlib histogram from a Results Column.
        Accepts as input a string column name.  Will extract the total
        entire column (if not already available) and plot it.

        See :meth:`marvin.utils.plot.scatter.hist` for details.

        Parameters:
            name (str):
                The name of the column of data. Required
            return_plateifus (bool):
                If True, includes the plateifus in each histogram bin in the
                histogram output.  Default is True.
            return_figure (bool):
                Set to False to not return the Figure and Axis object. Defaults to True.
            show_plot (bool):
                Set to False to not show the interactive plot
            **kwargs (dict):
                Any other keyword argument that will be passed to Marvin's
                hist plotting methods

        Returns:
            The histogram data, figure, and axes from the plotting function

        Example:
            >>> # do a query and get the results
            >>> q = Query(search_filter='nsa.z < 0.1', returnparams=['nsa.elpetro_ba', 'g_r'])
            >>> r = q.run()
            >>> # plot a histogram of the redshift column
            >>> hist_data, fig, axes = r.hist('nsa.z')

        '''

        return_plateifus = kwargs.pop('return_plateifus', True)
        show_plot = kwargs.pop('show_plot', True)
        return_figure = kwargs.get('return_figure', True)

        # get the named column
        col = self.columns[name]

        # get the values of the two columns
        if self.count != self.totalcount:
            data = self.getListOf(name, return_all=True)
        else:
            data = self.results[name]

        # xhist, fig, ax_hist_x = output
        with turn_off_ion(show_plot=show_plot):
            output = marvin.utils.plot.scatter.hist(data, **kwargs)

        if return_plateifus:
            plateifus = self.getListOf('plateifu', return_all=True)
            hdata = output[0] if return_figure else output
            hdata['bins_plateifu'] = map_bins_to_column(plateifus, hdata['indices'])
            output = (hdata,) + output[1:] if return_figure else hdata

        return output
