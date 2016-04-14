from __future__ import print_function
from marvin.tools.core import MarvinError, MarvinUserWarning
import warnings
import json
from functools import wraps
from astropy.table import Table, Column

__all__ = ['Results']


def local_mode_only(func):
    '''Decorator that bypasses function if in remote mode.'''

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.mode == 'remote':
            print('{0} not available in remote mode'.format(func.__name__))
        else:
            return func(self, *args, **kwargs)
    return wrapper


def remote_mode_only(func):
    '''Decorator that bypasses function if in local mode.'''

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.mode == 'local':
            print('{0} not available in local mode'.format(func.__name__))
        else:
            return func(self, *args, **kwargs)
    return wrapper


class Results(object):

    def __init__(self, results=None, query=None, count=None, mode=None):

        # super(Results, self).__init__(*args, **kwargs)
        self.results = results
        self.query = query
        self.count = count if count else len(results) if results else None
        self.mode = mode
        self.chunk = 10
        self.start = 0
        self.end = self.start + self.chunk

    def __repr__(self):
        return ('Results(results={0},\nquery={1},\ncount={2},\nmode={3})'
                .format(self.results, repr(self.query), self.count,
                        repr(self.mode)))

    def showQuery(self):
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
    def getDictOf(self, name='plateifu', format_type='listdict', to_json=False):
        ''' Get a dictionary of specified parameter '''

        # Test name in results
        output = None
        try:
            output = self.results[0].__getattribute__(name)
        except AttributeError as e:
            raise MarvinError('Name {0} not a property in results.  Try another: {1}'.format(name, e))

        # Format results
        if format_type == 'listdict':
            output = [{name: r.__getattribute__(name)} for r in self.results]
        elif format_type == 'dictlist':
            output = {name: [r.__getattribute__(name)for r in self.results]}
        else:
            output = None

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
