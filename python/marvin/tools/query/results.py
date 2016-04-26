from __future__ import print_function
from marvin.core import MarvinError, MarvinUserWarning
from marvin.tools.cube import Cube
import warnings
import json
import copy
from pprint import pformat
from functools import wraps
from astropy.table import Table, Column

__all__ = ['Results']


def local_mode_only(func):
    '''Decorator that bypasses function if in remote mode.'''

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.mode == 'remote':
            raise MarvinError('{0} not available in remote mode'.format(func.__name__))
        else:
            return func(self, *args, **kwargs)
    return wrapper


def remote_mode_only(func):
    '''Decorator that bypasses function if in local mode.'''

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.mode == 'local':
            raise MarvinError('{0} not available in local mode'.format(func.__name__))
        else:
            return func(self, *args, **kwargs)
    return wrapper


class Results(object):

    def __init__(self, *args, **kwargs):

        self.results = kwargs.get('results', None)
        self._queryobj = kwargs.get('queryobj', None)
        self.query = self._queryobj.query if self._queryobj else kwargs.get('query', None)
        self.returntype = self._queryobj.returntype if self._queryobj else kwargs.get('returntype', None)
        self.count = kwargs.get('count', None)
        self.mode = kwargs.get('mode', None)
        self.chunk = 10
        self.start = 0
        self.end = self.start + self.chunk

        # Auto convert to Marvin Object
        if self.returntype:
            self.convertToTool()

    def __repr__(self):
        # remote mode
        if isinstance(self.results, list):
            if len(self.results) > 5:
                data = self.results[:4]
                data.append('...')
                data.append('data'[-1])
                results = pformat(data)
            else:
                results = self.results
        # local mode
        else:
            out = {}
            for k in self.results:
                if k == 'data' and len(self.results['data']) > 5:
                    data = self.results['data'][:4]
                    data.append('...')
                    data.append(self.results['data'][-1])
                    out[k] = data
                else:
                    out[k] = self.results[k]
            results = pformat(out)
        return ('Results(results={0},\nquery={1},\ncount={2},\nmode={3})'
                .format(results, repr(self.query), self.count,
                        repr(self.mode)))

    def showQuery(self):
        ''' Displays the literal SQL query used to generate the Results objects
        '''
        return str(self.query.statement.compile(compile_kwargs={'literal_binds': True}))

    @local_mode_only
    def download(self):
        """Download data via sdsssync"""
        pass

    def sort(self):
        """WTForms Table? Bootstrap Table? dictionary?"""
        pass

    def toTable(self, columns=None):
        ''' Output the results as an astropy Table '''
        try:
            tabres = Table([self.results])
        except ValueError as e:
            raise MarvinError('Could not make astropy Table from results: {0}'.format(e))
        return tabres

    @remote_mode_only
    def toJson(self):
        ''' Output the results as a JSON object '''
        try:
            jsonres = json.dumps(self.results)
        except TypeError as e:
            raise MarvinError('Results not JSON-ifiable. Check the format of results: {0}'.format(e))
        return jsonres

    def getColumns(self):
        ''' Get the columns of the returned reults '''
        try:
            self.columns = self.results[0].keys() if self.results else None
        except Exception as e:
            raise MarvinError('Could not get table keys from results.  Results not an SQLalchemy results collection: {0}'.format(e))
        return self.columns

    @local_mode_only
    def getListOf(self, name='plateifu', to_json=False):
        ''' Get a list of plate-IFUs or MaNGA IDs from results '''

        output = None
        try:
            output = [r.__getattribute__(name) for r in self.results]
        except AttributeError as e:
            raise MarvinError('Name {0} not a property in results.  Try another: {1}'.format(name, e))

        if to_json:
            output = json.dumps(output) if output else None

        return output

    @local_mode_only
    def getDictOf(self, name=None, format_type='listdict', to_json=False):
        ''' Get a dictionary of specified parameter '''

        # Try to get the sqlalchemy results keys
        keys = self.getColumns()

        # Format results
        if format_type == 'listdict':
            output = [{k: res.__getattribute__(k) for k in keys} for res in self.results]
        elif format_type == 'dictlist':
            output = {k: [res.__getattribute__(k) for res in self.results] for k in keys}
        else:
            raise MarvinError('No output.  Check your input format_type.')

        # Test if name is in results
        if name:
            nameinkeys = name in keys if keys and name else None

            if nameinkeys:
                # Format results
                if format_type == 'listdict':
                    output = [{name: i[name]} for i in output]
                elif format_type == 'dictlist':
                    output = output[name]
                else:
                    output = None
            else:
                raise MarvinError('Name {0} not a property in results.  Try another'.format(name))

        if to_json:
            output = json.dumps(output) if output else None

        return output

    @local_mode_only
    def getNext(self, chunk=None):
        ''' Get the next set of results from the query, from start to end in units of chunk '''

        newstart = self.end
        self.chunk = chunk if chunk else self.chunk
        newend = newstart + self.chunk
        if newend > self.count:
            warnings.warn('You have reached the end.', MarvinUserWarning)
            newend = self.count
            newstart = newend - self.chunk

        print('Retrieving next {0}, from {1} to {2}'.format(self.chunk, newstart, newend))
        self.results = self.query.slice(newstart, newend).all()
        self.start = newstart
        self.end = newend

        return self.results

    @local_mode_only
    def getPrevious(self, chunk=None):
        ''' Get the previous set of results from the query, from start to end in units of chunk '''

        newend = self.start
        self.chunk = chunk if chunk else self.chunk
        newstart = newend - self.chunk
        if newstart < 0:
            warnings.warn('You have reached the beginning.', MarvinUserWarning)
            newstart = 0
            newend = newstart + self.chunk

        print('Retrieving previous {0}, from {1} to {2}'.format(self.chunk, newstart, newend))
        self.results = self.query.slice(newstart, newend).all()
        self.start = newstart
        self.end = newend

        return self.results

    @local_mode_only
    def getSubset(self, start, limit=10):
        ''' Gets a slice of set of results '''
        start = 0 if start < 0 else int(start)
        end = start + int(limit)
        # if end > self.count:
        #     end = self.count
        #     start = end - int(limit)
        self.start = start
        self.end = end
        self.chunk = limit
        self.results = self.query.slice(start, end).all()
        return self.results

    @local_mode_only
    def getAll(self):
        ''' Retrieve all of the results '''
        self.results = self.query.all()
        return self.results

    def convertToTool(self):
        ''' Converts the list of results into a Marvin Tool object '''
        if self.returntype == 'cube':
            self.objects = [Cube(mangaid=res.__getattribute__('mangaid'), mode=self.mode) for res in self.results]

