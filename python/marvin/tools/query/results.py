from __future__ import print_function
from marvin.tools.core import MarvinToolsClass, MarvinError, MarvinUserWarning
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

    def getAll(self):
        ''' Retrieve all of the results '''
        self.results = self.query.all()
