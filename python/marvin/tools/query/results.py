from __future__ import print_function
from marvin.tools.core import MarvinError, MarvinUserWarning
import warnings
import json
from astropy.table import Table, Column

__all__ = ['Results']


class Results(object):

    def __init__(self, results=None, query=None, count=None):

        # super(Results, self).__init__(*args, **kwargs)
        self.results = results
        self.query = query
        self.count = count if count else len(results) if results else None
        self.chunk = 10
        self.start = 0
        self.end = self.start + self.chunk

    def showQuery(self):
        return str(self.query.statement.compile(compile_kwargs={'literal_binds': True}))

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

    def toJson(self):
        ''' Output the results as a JSON object '''
        try:
            jsonres = json.dumps(self.results)
        except TypeError as e:
            raise MarvinError('Results not JSON-ifiable. Check the format of results: {0}'.format(e))
        return jsonres

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

    def getAll(self):
        ''' Retrieve all of the results '''
        self.results = self.query.all()
        return self.results
