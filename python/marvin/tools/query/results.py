from __future__ import print_function
from marvin.tools.core import MarvinToolsClass, MarvinError, MarvinUserWarning
import warnings

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
        pass

    def toTable(self):
        pass

    def toJson(self):
        pass

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
        self.results = self.query.all()
