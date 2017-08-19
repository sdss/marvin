from __future__ import print_function
from marvin.core.exceptions import MarvinError, MarvinUserWarning, MarvinBreadCrumb
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.rss import RSS
from marvin.tools.modelcube import ModelCube
from marvin.tools.spaxel import Spaxel
from marvin.tools.query.forms import MarvinForm
from marvin.tools.query.query_utils import ParameterGroup
from marvin import config, log
from marvin.utils.general import getImagesByList, downloadList, parseIdentifier
from marvin.api.api import Interaction
from operator import add
import warnings
import json
import os
import datetime
import numpy as np
import six
from fuzzywuzzy import process
from functools import wraps
from astropy.table import Table, vstack, hstack
from collections import OrderedDict, namedtuple
from marvin.core import marvin_pickle

try:
    import cPickle as pickle
except:
    import pickle

try:
    import pandas as pd
except ImportError:
    warnings.warn('Could not import pandas.', MarvinUserWarning)

__all__ = ['Results']

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


def marvintuple(name, params=None, **kwargs):
    ''' Custom namedtuple class factory for Marvin Results rows

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

    '''

    # check the params input
    if params and isinstance(params, six.string_types):
        params = params.split(',') if ',' in params else [params]
        params = [p.strip() for p in params]

    # pop any extra keywords
    results = kwargs.pop('results', None)

    # create default namedtuple and find new columns
    default = namedtuple(name, 'mangaid, plate, plateifu, ifu_name')
    newcols = [col for col in params if col not in default._fields] if params else None
    finalfields = default._fields + tuple(newcols) if newcols else default._fields
    nt = namedtuple(name, finalfields, **kwargs)

    def new_add(self, other):
        ''' Overloaded add to combine tuples without duplicates '''

        if self._release:
            assert self._release == other._release, 'Cannot add result rows from different releases'
        if self._searchfilter:
            assert self._searchfilter == other._searchfilter, ('Cannot add result rows generated '
                                                               'using different search filters')

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
    nt._release = results._release if results else None
    nt._searchfilter = results.searchfilter if results else None
    return nt


class ResultSet(list):
    ''' A Set of Results '''
    def __init__(self, _objects, **kwargs):
        list.__init__(self, _objects)
        self.count = kwargs.get('count', None)
        self.total = kwargs.get('total', None)
        self.pages = int(np.ceil(self.total / float(self.count)))
        self._results = kwargs.get('results', None)
        self.columns = kwargs.get('columns', None)
        self.index = kwargs.get('index') if kwargs.get('index') else 0
        self.current_page = (int(self.index) + self.count) / self.count
        if self._results:
            self.columns = self._results.columns if not self.columns else self.columns
            self.choices = self.columns.list_params(remote=True)

    def __repr__(self):
        old = list.__repr__(self)
        return ('<ResultSet(set={0}/{1}, count_in_set={2}, '
                'total={3})>\n{4}'.format(self.current_page, self.pages, self.count,
                                          self.total, old.replace('),', '),\n')))

    def __getitem__(self, value):
        if isinstance(value, six.string_types):
            if value in self.columns:
                colname = self.columns[value].remote
                rows = [row.__getattribute__(colname) for row in self]
            else:
                rows = [row for row in self if value in row]
            if rows:
                return rows[0] if len(rows) == 1 else rows
            else:
                raise ValueError('{0} not found in the list'.format(value))
        else:
            return list.__getitem__(self, value)

    def __add__(self, other):
        newcols = self.columns.full + [col.full for col in other.columns if col.full not in self.columns.full]
        newcols = ParameterGroup('Columns', [{'full': p} for p in newcols])
        newresults = self._results
        newresults.columns = newcols
        newset = map(add, self, other)
        return ResultSet(newset, count=self.count, total=self.total, index=self.index,
                         columns=newcols, results=newresults)


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
            returntype (str):
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

        Returns:
            results: An object representing the Results entity

        Example:
            >>> f = 'nsa.z < 0.012 and ifu.name = 19*'
            >>> q = Query(searchfilter=f)
            >>> r = q.run()
            >>> print(r)
            >>> Results(results=[(u'4-3602', u'1902', -9999.0), (u'4-3862', u'1902', -9999.0), (u'4-3293', u'1901', -9999.0), (u'4-3988', u'1901', -9999.0), (u'4-4602', u'1901', -9999.0)],
            >>>         query=<sqlalchemy.orm.query.Query object at 0x115217090>,
            >>>         count=64,
            >>>         mode=local)

    '''

    def __init__(self, *args, **kwargs):

        self.results = kwargs.get('results', None)
        self._queryobj = kwargs.get('queryobj', None)
        self._updateQueryObj(**kwargs)
        self.count = kwargs.get('count', None)
        self.totalcount = kwargs.get('totalcount', self.count)
        self._runtime = kwargs.get('runtime', None)
        self.query_time = self._getRunTime() if self._runtime is not None else None
        self.response_time = kwargs.get('response_time', None)
        self.mode = config.mode if not kwargs.get('mode', None) else kwargs.get('mode', None)
        self.chunk = self.limit if self.limit else kwargs.get('chunk', 100)
        self.start = kwargs.get('start', 0)
        self.end = kwargs.get('end', self.start + self.chunk)
        self.coltoparam = None
        self.paramtocol = None
        self.objects = None
        self.sortcol = None
        self.order = None

        # drop breadcrumb
        breadcrumb.drop(message='Initializing MarvinResults {0}'.format(self.__class__),
                        category=self.__class__)

        # Convert remote results to NamedTuple
        #if self.mode == 'remote':
        self._makeNamedTuple()

        # Setup columns, and parameters
        #if self.count > 0:
        #    self._setupColumns()

        # Auto convert to Marvin Object
        if self.returntype:
            self.convertToTool(self.returntype)

    def _updateQueryObj(self, **kwargs):
        ''' update parameters using the _queryobj '''
        self.query = self._queryobj.query if self._queryobj else kwargs.get('query', None)
        self.returntype = self._queryobj.returntype if self._queryobj else kwargs.get('returntype', None)
        self.searchfilter = self._queryobj.searchfilter if self._queryobj else kwargs.get('searchfilter', None)
        self.returnparams = self._queryobj._returnparams if self._queryobj else kwargs.get('returnparams', None)
        self.limit = self._queryobj.limit if self._queryobj else kwargs.get('limit', None)
        self._params = self._queryobj.params if self._queryobj else kwargs.get('params', None)
        self._release = self._queryobj._release if self._queryobj else kwargs.get('release', None)

    def __repr__(self):
        return ('Marvin Results(query={0}, totalcount={1}, count={2}, mode={3})'.format(self.searchfilter, self.totalcount, self.count, self.mode))

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

            Returns:
                NA: na

            Example:
                >>> r = q.run()
                >>> r.returntype = 'cube'
                >>> r.download()
        '''

        plates = self.getListOf(name='plate')
        ifus = self.getListOf(name='ifu.name')
        plateifu = ['{0}-{1}'.format(z[0], z[1]) for z in zip(plates, ifus)]
        if images:
            tmp = getImagesByList(plateifu, mode='remote', as_url=True, download=True)
        else:
            downloadList(plateifu, dltype=self.returntype, limit=limit)

    def sort(self, name, order='asc'):
        ''' Sort the set of results by column name

            Sorts the results by a given parameter / column name.  Sets
            the results to the new sorted results.

            Parameters:
                name (str):
                order ({'asc', 'desc'}):

            Returns:
                sortedres (list):
                    The listed of sorted results.

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
        refname = self._getRefName(name)
        self.sortcol = refname
        self.order = order

        if self.mode == 'local':
            reverse = True if order == 'desc' else False
            sortedres = sorted(self.results, key=lambda row: row.__getattribute__(refname), reverse=reverse)
            self.results = sortedres
        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['querycubes']['url']
            params = {'searchfilter': self.searchfilter, 'params': self.returnparams,
                      'sort': refname, 'order': order, 'limit': self.limit}
            self._interaction(url, params, named_tuple=True, calltype='Sort')
            sortedres = self.results

        return sortedres

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
            tabres = Table(rows=self.results, names=self.paramtocol.keys())
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
            >>> q, r = doQuery(searchfilter='nsa.z < 0.1', returnparams=['g_r', 'cube.ra', 'cube.dec'])
            >>> # query 2
            >>> q2, r2 = doQuery(searchfilter='nsa.z < 0.1')
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

    def toFits(self, filename='myresults.fits'):
        ''' Output the results as a FITS file

        Writes a new FITS file from search results using
        the astropy Table.write()

        Parameters:
            filename (str):
                Name of FITS file to output

        '''
        myext = os.path.splitext(filename)[1]
        if not myext:
            filename = filename + '.fits'
        table = self.toTable()
        table.write(filename, format='fits')
        print('Writing new FITS file {0}'.format(filename))

    def toCSV(self, filename='myresults.csv'):
        ''' Output the results as a CSV file

        Writes a new CSV file from search results using
        the astropy Table.write()

        Parameters:
            filename (str):
                Name of CSV file to output

        '''
        myext = os.path.splitext(filename)[1]
        if not myext:
            filename = filename + '.csv'
        table = self.toTable()
        table.write(filename, format='csv')
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
            dfres = pd.DataFrame(self.results)
        except (ValueError, NameError) as e:
            raise MarvinError('Could not make pandas dataframe from results: {0}'.format(e))
        return dfres

    def _makeNamedTuple(self, index=None):
        ''' '''
        #ntnames = self._reduceNames(self._params, under=True)
        # try:
        #     nt = namedtuple('MarvinRow', ntnames)
        # except ValueError as e:
        #     raise MarvinError('Cannot created MarvinRow from remote Results: {0}'.format(e))
        # else:
        #     globals()[nt.__name__] = nt

        # qpo = ntnames

        # def keys(self):
        #     return qpo
        # nt.keys = keys
        #if 'name' in ntnames:
        #    ntnames[ntnames.index('name')] = 'ifu_name'

        self.columns = self.getColumns()
        ntnames = self.columns.list_params(remote=True)
        nt = marvintuple('ResultRow', ntnames, results=self)
        results = [nt(*r) for r in self.results]
        self.count = len(self.results)
        self.results = ResultSet(results, count=self.count, total=self.totalcount, index=index, results=self)

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

        # set Marvin query object to None, in theory this could be pickled as well
        self._queryobj = None
        isnotstr = not isinstance(self.query, six.string_types)
        if isnotstr:
            self.query = None

        sf = self.searchfilter.replace(' ', '') if self.searchfilter else 'anon'
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

        try:
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
        return marvin_pickle.restore(path, delete=delete)

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
        ''' Get the column names of the returned reults

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
            #self.columns = list(self.results[0]._fields) if self.results else None
            self.columns = ParameterGroup('Columns', [{'full': p} for p in self._params])
        except Exception as e:
            raise MarvinError('Could not create query columns: {0}'.format(e))
        return self.columns

    # def _reduceNames(self, columns, under=None):
    #     ''' reduces the full table.parameter names to non-dotted versions '''
    #     # fullstr = ','.join(columns)
    #     # names = [t if '.' not in t else t.split('.')[1] if fullstr.count(
    #     #             '.'+t.split('.')[1]) == 1 else t for t in columns]
    #     colsplit = [c if '.' not in c else c.split('.')[1] for c in columns]
    #     names = [t if '.' not in t else t.split('.')[1] if colsplit.count(
    #         t.split('.')[1]) == 1 else t for t in columns]
    #     # replace . with _
    #     if under:
    #         names = [n if '.' not in n else n.replace('.', '_')for n in names]
    #     return names

    # def _setupColumns(self):
    #     ''' Auto sets up all the column/parameter name info '''
    #     columns = self.getColumns()
    #     redcol = self._reduceNames(columns)
    #     tmp = self.mapColumnsToParams(inputs=redcol)
    #     tmp = self.mapParamsToColumns(inputs=redcol)

    # def mapColumnsToParams(self, col=None, inputs=None):
    #     ''' Map the columns names from results to the original parameter names '''
    #     columns = self.getColumns()
    #     #params = self._params if self.mode == 'local' else self._params
    #     cols = columns if not inputs else inputs
    #     if cols:
    #         if not self.coltoparam:
    #             self.coltoparam = OrderedDict(zip(cols, self._params))
    #         mapping = self.coltoparam[col] if col else list(self.coltoparam.values())
    #     else:
    #         mapping = None
    #     return mapping

    # def mapParamsToColumns(self, param=None, inputs=None):
    #     ''' Map original parameter names to the column names '''
    #     columns = self.getColumns()
    #     #params = self._params if self.mode == 'local' else self._params
    #     cols = columns if not inputs else inputs
    #     if cols:
    #         if not self.paramtocol:
    #             self.paramtocol = OrderedDict(zip(self._params, cols))
    #         mapping = self.paramtocol[param] if param else list(self.paramtocol.values())
    #     else:
    #         mapping = None
    #     return mapping

    def _interaction(self, url, params, calltype='', named_tuple=None):
        ''' Perform a remote Interaction call

        Parameters:
            url (str):
                The url of the request
            params (dict):
                A dictionary of parameters (get or post) to send with the request
            calltype (str):
                The method call sending the request
            named_tuple (bool):
                If True, sets the response output as the new results and creates a
                new named tuple set

        Returns:
            output:
                The output data from the request

        Raises:
            MarvinError: Raises on any HTTP Request error

        '''

        # check if the returnparams parameter is in the proper format
        if 'params' in params:
            return_params = params.get('params')
            if isinstance(return_params, list):
                params['params'] = ','.join(return_params)

        # send the request
        try:
            ii = Interaction(route=url, params=params)
        except MarvinError as e:
            raise MarvinError('API Query {0} call failed: {1}'.format(calltype, e))
        else:
            output = ii.getData()
            self.response_time = ii.response_time
            self._runtime = ii.results['runtime']
            self.query_time = self._getRunTime()
            if named_tuple:
                self.results = output
                self._makeNamedTuple()
            else:
                return output

    def _check_column(self, name):
        ''' Check if a name exists as a column '''
        try:
            name_in_col = name in self.columns
        except KeyError as e:
            raise MarvinError('Column {0} not found in results: {1}'.format(name, e))

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
        self._check_column()
        fullname = self.columns[name].full

        # deal with the output
        if return_all:
            # grab all of that column
            if self.mode == 'local':
                results = self._queryobj._query_column(fullname)
                output = list(sum(results, ()))
            elif self.mode == 'remote':
                # Get the query route
                url = config.urlmap['api']['getcolumn']['url'].format(colname=fullname)
                params = {'searchfilter': self.searchfilter, 'format_type': 'list'}
                output = self._interaction(url, params, calltype='getList')
        else:
            # only deal with current page
            output = self.results[name]

        if to_json:
            output = json.dumps(output) if output else None

        if to_ndarray:
            output = np.array(output) if output else None

        return output

    def getDictOf(self, name=None, format_type='listdict', to_json=False):
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

        # Try to get the sqlalchemy results keys
        keys = self.getColumns()
        tmp = self.mapColumnsToParams()
        tmp = self.mapParamsToColumns()

        if self.mode == 'local':
            lookup = OrderedDict(zip(keys, keys))
        elif self.mode == 'remote':
            lookup = self.coltoparam

        # Format results
        if format_type == 'listdict':
            output = [{lookup[k]: res.__getattribute__(k) for k in keys} for res in self.results]
        elif format_type == 'dictlist':
            output = {lookup[k]: [res.__getattribute__(k) for res in self.results] for k in keys}
        else:
            raise MarvinError('No output.  Check your input format_type.')

        # Test if name is in results
        if name:
            refname = self._getRefName(name)
            if refname:
                # Format results
                newname = lookup[refname]
                if format_type == 'listdict':
                    output = [{newname: i[newname]} for i in output]
                elif format_type == 'dictlist':
                    output = {newname: output[newname]}
                else:
                    output = None
            else:
                raise MarvinError('Name {0} not a property in results.  Try another'.format(name))

        if to_json:
            output = json.dumps(output) if output else None

        return output

    def _getRefName(self, name):
        ''' Get the appropriate reference column / parameter name

        This converts an input name into the respective column name
        '''
        cols = self.getColumns()

        if name in cols:
            # name already in the list of column names
            refname = name
        else:
            if not self.coltoparam:
                pars = self.mapColumnsToParams()
            if not self.paramtocol:
                tmp = self.mapParamsToColumns()

            refname = self.coltoparam[name] if name in self.coltoparam else \
                self.paramtocol[name] if name in self.paramtocol else None

        return refname

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
        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['getsubset']['url']
            params = {'searchfilter': self.searchfilter, 'params': self.returnparams,
                      'start': newstart, 'end': newend, 'limit': self.limit,
                      'sort': self.sortcol, 'order': self.order}
            self._interaction(url, params, calltype='getNext', named_tuple=True)

        self.start = newstart
        self.end = newend
        self.count = len(self.results)

        if self.returntype:
            self.convertToTool()

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
        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['getsubset']['url']

            params = {'searchfilter': self.searchfilter, 'params': self.returnparams,
                      'start': newstart, 'end': newend, 'limit': self.limit,
                      'sort': self.sortcol, 'order': self.order}
            self._interaction(url, params, calltype='getPrevious', named_tuple=True)

        self.start = newstart
        self.end = newend
        self.count = len(self.results)

        if self.returntype:
            self.convertToTool()

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
        '''

        if not limit:
            limit = self.chunk

        if limit < 0:
            warnings.warn('Limit cannot be negative. Setting to {0}'.format(self.chunk), MarvinUserWarning)
            limit = self.chunk

        start = 0 if int(start) < 0 else int(start)
        end = start + int(limit)
        # if end > self.count:
        #     end = self.count
        #     start = end - int(limit)
        self.start = start
        self.end = end
        self.chunk = limit
        if self.mode == 'local':
            self.results = self.query.slice(start, end).all()
        elif self.mode == 'remote':
            # Fail if no route map initialized
            if not config.urlmap:
                raise MarvinError('No URL Map found.  Cannot make remote call')

            # Get the query route
            url = config.urlmap['api']['getsubset']['url']

            params = {'searchfilter': self.searchfilter, 'params': self.returnparams,
                      'start': start, 'end': end, 'limit': self.limit,
                      'sort': self.sortcol, 'order': self.order}
            self._interaction(url, params, calltype='getSubset', named_tuple=True)

        self.count = len(self.results)
        if self.returntype:
            self.convertToTool()

        return self.results

    def merge_subsets(self, subsets):
        ''' Merges a list of subsets together into a single list '''

        isnested = all(isinstance(i, list) for i in subsets)
        if not isnested:
            raise MarvinError('Input must be a list of result subsets (list of lists)')

        # check that subsets have the same number of columns and same column names
        same_count = len(set([len(s[0]) for s in subsets]))
        same_cols = all([list(s[0]._fields) == self.columns for s in subsets])

        # merge subsets
        if same_count:
            if same_cols:
                # combine into one set
                output = sum(subsets, [])
                print('Setting results to new subset of size {0}.'.format(len(output)))
                self.results = output
                self.count = len(self.results)
                return self.results
            else:
                raise MarvinUserWarning('Cannot merge subsets. Different named columns from base set. '
                                        'Ensure all columns are the same.')
        else:
            raise MarvinUserWarning('Cannot merge subsets. Column number mismatch among subsets. '
                                    'Ensure all subsets have the same number of columns')

    def combine_columns(self, results):
        ''' Combine columns from multiple Marvin Results objects '''

        # make results a list if it is not
        if not isinstance(results, list):
            results = [results]

        # check if all items in the results list are of Marvin Results type
        isnested = all(isinstance(i, Results) for i in results)
        if not isnested:
            raise MarvinError('Input must be a list of Marvin Results objects')

        # check for same search criteria
        same_search = all([i.searchfilter == self.searchfilter for i in results])
        if not same_search:
            raise MarvinUserWarning('Cannot merge columns. The seach filter in your Marvin '
                                    'Results do not match with each other or the base set')

        # check for consistency between Results
        plateifus = self.getListOf('plateifu')
        same_rows = all([i.getListOf('plateifu') == plateifus for i in results])
        if not same_rows:
            raise MarvinUserWarning('Cannot merge columns. The rows in your Marvin '
                                    'Results do not match with each other or the base set')

        # get new columns
        all_columns = (sum([i._params for i in results], []))
        all_columns = list(OrderedDict.fromkeys(all_columns))
        new_columns = set(all_columns) - set(self._params)

        kwargs = {'results': 'new', 'count': self.count, 'mode': self.mode, 'queryobj': 'new',
                  'totalcount': self.totalcount, 'chunk': self.limit, 'runtime': self.runtime}
        return Results(**kwargs)
        # Results(results=res, query=self.query, mode=self.mode, queryobj=self, count=count,
        #                    returntype=self.returntype, totalcount=totalcount, chunk=chunk,
        #                    runtime=query_runtime, response_time=resp_runtime)

    #@local_mode_only
    def getAll(self):
        ''' Retrieve all of the results of a query

            Attempts to return all the results of a query.  The efficiency of this
            method depends heavily on how many rows and columns you wish to return.

            A cutoff limit is applied for results with more than 500,000 rows or
            results with more than 25 columns.

            Returns:
                The full list of query results.
        '''

        if self.totalcount > 500000 or len(self.columns) > 25:
            raise MarvinUserWarning("Cannot retrieve all results. The total number of requested "
                                    "rows or columns is too high. Please use the getSubset "
                                    "method retrieve pages.")

        if self.mode == 'local':
            self.results = self.query.all()
            self._makeNamedTuple()
        elif self.mode == 'remote':
            # Get the query route
            url = config.urlmap['api']['querycubes']['url']

            params = {'searchfilter': self.searchfilter, 'return_all': True}
            self._interaction(url, params, calltype='getAll', named_tuple=True)
            self.count = self.totalcount
            print('Returned all {0} results'.format(self.totalcount))

    def convertToTool(self, tooltype, **kwargs):
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
        mode = kwargs.get('mode', self.mode)
        limit = kwargs.get('limit', None)
        toollist = ['cube', 'spaxel', 'maps', 'rss', 'modelcube']
        tooltype = tooltype if tooltype else self.returntype
        assert tooltype in toollist, 'Returned tool type must be one of {0}'.format(toollist)

        # get the parameter list to check against
        paramlist = self.paramtocol.keys()

        print('Converting results to Marvin {0} objects'.format(tooltype.title()))
        if tooltype == 'cube':
            self.objects = [Cube(plateifu=res.__getattribute__(
                            self._getRefName('cube.plateifu')),
                            mode=mode) for res in self.results[0:limit]]
        elif tooltype == 'maps':

            isbin = 'bintype.name' in paramlist
            istemp = 'template.name' in paramlist
            self.objects = []

            for res in self.results[0:limit]:
                mapkwargs = {'mode': mode, 'plateifu': res.__getattribute__(self._getRefName('cube.plateifu'))}

                if isbin:
                    binval = res.__getattribute__(self._getRefName('bintype.name'))
                    mapkwargs['bintype'] = binval

                if istemp:
                    tempval = res.__getattribute__(self._getRefName('template.name'))
                    mapkwargs['template_kin'] = tempval

                self.objects.append(Maps(**mapkwargs))
        elif tooltype == 'spaxel':

            assert 'spaxelprop.x' in paramlist and 'spaxelprop.y' in paramlist, \
                   'Parameters must include spaxelprop.x and y in order to convert to Marvin Spaxel.'

            self.objects = []

            load = kwargs.get('load', True)
            maps = kwargs.get('maps', True)
            modelcube = kwargs.get('modelcube', True)

            tab = self.toTable()
            uniq_plateifus = list(set(self.getListOf('plateifu')))

            for plateifu in uniq_plateifus:
                c = Cube(plateifu=plateifu, mode=mode)
                univals = tab['cube.plateifu'] == plateifu
                x = tab[univals]['spaxelprop.x'].tolist()
                y = tab[univals]['spaxelprop.y'].tolist()
                spaxels = c[y, x]
                self.objects.extend(spaxels)

            # for res in self.results[0:limit]:
            #     spaxkwargs = {'plateifu': res.__getattribute__(self._getRefName('cube.plateifu')),
            #                   'x': res.__getattribute__(self._getRefName('spaxelprop.x')),
            #                   'y': res.__getattribute__(self._getRefName('spaxelprop.y')),
            #                   'load': load, 'maps': maps, 'modelcube': modelcube}
            #     self.objects.append(Spaxel(**spaxkwargs))
        elif tooltype == 'rss':
            self.objects = [RSS(plateifu=res.__getattribute__(
                            self._getRefName('cube.plateifu')),
                            mode=mode) for res in self.results[0:limit]]
        elif tooltype == 'modelcube':

            isbin = 'bintype.name' in paramlist
            istemp = 'template.name' in paramlist
            self.objects = []

            for res in self.results[0:limit]:
                mapkwargs = {'mode': mode, 'plateifu': res.__getattribute__(self._getRefName('cube.plateifu'))}

                if isbin:
                    binval = res.__getattribute__(self._getRefName('bintype.name'))
                    mapkwargs['bintype'] = binval

                if istemp:
                    tempval = res.__getattribute__(self._getRefName('template.name'))
                    mapkwargs['template_kin'] = tempval

                self.objects.append(ModelCube(**mapkwargs))

