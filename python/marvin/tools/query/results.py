from __future__ import print_function
from marvin.core.exceptions import MarvinError, MarvinUserWarning, MarvinBreadCrumb
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
from marvin.tools.rss import RSS
from marvin.tools.modelcube import ModelCube
from marvin.tools.spaxel import Spaxel
from marvin import config, log
from marvin.utils.general import getImagesByList, downloadList
from marvin.api.api import Interaction
import warnings
import json
import os
import datetime
from functools import wraps
from astropy.table import Table
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
        self.query_runtime = self._getRunTime() if self._runtime is not None else None
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
        if self.mode == 'remote':
            self._makeNamedTuple()

        # Setup columns, and parameters
        if self.count > 0:
            self._setupColumns()

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

    def __repr__(self):
        return ('Marvin Results(results={0}, \nquery={1}, \ncount={2}, \nmode={3}'.format(self.results[0], repr(self.query), self.count, self.mode))

    def showQuery(self):
        ''' Displays the literal SQL query used to generate the Results objects

            Returns:
                querystring (str):
                    A string representation of the SQL query
        '''

        # check unicode or str
        try:
            isstr = type(self.query) == unicode
        except NameError as e:
            isstr = type(self.query) == str

        # return the string query or compile the real query
        if isstr:
            return self.query
        else:
            return str(self.query.statement.compile(compile_kwargs={'literal_binds': True}))

    def _getRunTime(self):
        ''' Sets the query runtime as a datetime timedelta object '''
        if type(self._runtime) == dict:
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
            tmp = downloadList(plateifu, dltype=self.returntype, limit=limit)

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
            try:
                ii = Interaction(route=url, params=params)
            except MarvinError as e:
                raise MarvinError('API Query Sort call failed: {0}'.format(e))
            else:
                self.results = ii.getData()
                self._makeNamedTuple()
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
            tabres = Table(rows=self.results, names=self.getColumns())
        except ValueError as e:
            raise MarvinError('Could not make astropy Table from results: {0}'.format(e))
        return tabres

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
            filename = filename+'.fits'
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
            filename = filename+'.csv'
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

    def _makeNamedTuple(self):
        ''' '''
        ntnames = self._reduceNames(self._params, under=True)
        try:
            nt = namedtuple('NamedTuple', ntnames)
        except ValueError as e:
            raise MarvinError('Cannot created NamedTuple from remote Results: {0}'.format(e))
        else:
            globals()[nt.__name__] = nt

        qpo = ntnames

        def keys(self):
            return qpo
        nt.keys = keys
        self.results = [nt(*r) for r in self.results]

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
        try:
            isnotstr = type(self.query) != unicode
        except NameError as e:
            isnotstr = type(self.query) != str
        if isnotstr:
            self.query = None

        sf = self.searchfilter.replace(' ', '') if self.searchfilter else 'anon'
        # set the path
        if not path:
            path = os.path.expanduser('~/marvin_results_{0}.mpf'.format(sf))

        # check for file extension
        if not os.path.splitext(path)[1]:
            path = os.path.join(path+'.mpf')

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

            Parameters:
                None

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
            self.columns = list(self.results[0].keys()) if self.results else None
        except Exception as e:
            raise MarvinError('Could not get table keys from results.  Results not an SQLalchemy results collection: {0}'.format(e))
        return self.columns

    def _reduceNames(self, columns, under=None):
        ''' reduces the full table.parameter names to non-dotted versions '''
        # fullstr = ','.join(columns)
        # names = [t if '.' not in t else t.split('.')[1] if fullstr.count(
        #             '.'+t.split('.')[1]) == 1 else t for t in columns]
        colsplit = [c if '.' not in c else c.split('.')[1] for c in columns]
        names = [t if '.' not in t else t.split('.')[1] if colsplit.count(
                    t.split('.')[1]) == 1 else t for t in columns]
        # replace . with _
        if under:
            names = [n if '.' not in n else n.replace('.', '_')for n in names]
        return names

    def _setupColumns(self):
        ''' Auto sets up all the column/parameter name info '''
        columns = self.getColumns()
        redcol = self._reduceNames(columns)
        tmp = self.mapColumnsToParams(inputs=redcol)
        tmp = self.mapParamsToColumns(inputs=redcol)

    def mapColumnsToParams(self, col=None, inputs=None):
        ''' Map the columns names from results to the original parameter names '''
        columns = self.getColumns()
        #params = self._params if self.mode == 'local' else self._params
        cols = columns if not inputs else inputs
        if cols:
            if not self.coltoparam:
                self.coltoparam = OrderedDict(zip(cols, self._params))
            mapping = self.coltoparam[col] if col else list(self.coltoparam.values())
        else:
            mapping = None
        return mapping

    def mapParamsToColumns(self, param=None, inputs=None):
        ''' Map original parameter names to the column names '''
        columns = self.getColumns()
        #params = self._params if self.mode == 'local' else self._params
        cols = columns if not inputs else inputs
        if cols:
            if not self.paramtocol:
                self.paramtocol = OrderedDict(zip(self._params, cols))
            mapping = self.paramtocol[param] if param else list(self.paramtocol.values())
        else:
            mapping = None
        return mapping

    def getListOf(self, name=None, to_json=False):
        ''' Extract a list of a single parameter from results

            Parameters:
                name (str):
                    Name of the parameter name to return.  If not specified,
                    it returns all parameters.
                to_json (bool):
                    True/False boolean to convert the output into a JSON format

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

        # get reference name
        refname = self._getRefName(name)
        if not refname:
            raise MarvinError('Name {0} not a property in results.  Try another.'.format(refname))

        output = None
        try:
            output = [r.__getattribute__(refname) for r in self.results]
        except AttributeError as e:
            raise MarvinError('Name {0} not a property in results.  Try another: {1}'.format(refname, e))

        if to_json:
            output = json.dumps(output) if output else None

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
            print('refname', refname)
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

        newstart = self.end
        self.chunk = chunk if chunk else self.chunk
        newend = newstart + self.chunk
        if newend > self.totalcount:
            warnings.warn('You have reached the end.', MarvinUserWarning)
            newend = self.totalcount
            newstart = newend - self.chunk

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
            try:
                ii = Interaction(route=url, params=params)
            except MarvinError as e:
                raise MarvinError('API Query GetNext call failed: {0}'.format(e))
            else:
                self.results = ii.getData()
                self._makeNamedTuple()

        self.start = newstart
        self.end = newend

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

        newend = self.start
        self.chunk = chunk if chunk else self.chunk
        newstart = newend - self.chunk
        if newstart < 0:
            warnings.warn('You have reached the beginning.', MarvinUserWarning)
            newstart = 0
            newend = newstart + self.chunk

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
            try:
                ii = Interaction(route=url, params=params)
            except MarvinError as e:
                raise MarvinError('API Query GetNext call failed: {0}'.format(e))
            else:
                self.results = ii.getData()
                self._makeNamedTuple()

        self.start = newstart
        self.end = newend

        if self.returntype:
            self.convertToTool()

        return self.results

    def getSubset(self, start, limit=10):
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
            try:
                ii = Interaction(route=url, params=params)
            except MarvinError as e:
                raise MarvinError('API Query GetNext call failed: {0}'.format(e))
            else:
                self.results = ii.getData()
                self._makeNamedTuple()

        if self.returntype:
            self.convertToTool()

        return self.results

    @local_mode_only
    def getAll(self):
        ''' Retrieve all of the results of a query

            Parameters:
                None

            Returns:
                results (list):
                    A list of query results.
        '''
        self.results = self.query.all()
        return self.results

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

            for res in self.results[0:limit]:
                spaxkwargs = {'plateifu': res.__getattribute__(self._getRefName('cube.plateifu')),
                              'x': res.__getattribute__(self._getRefName('spaxelprop.x')),
                              'y': res.__getattribute__(self._getRefName('spaxelprop.y')),
                              'load': load, 'maps': maps, 'modelcube': modelcube}
                self.objects.append(Spaxel(**spaxkwargs))
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

