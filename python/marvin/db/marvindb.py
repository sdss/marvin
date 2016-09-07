#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-03-28 23:30:14
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-03-28 23:30:14 by Brian Cherinka
    Last Modified On: 2016-03-28 23:30:14 by Brian

'''
from __future__ import print_function
from __future__ import division
from brain.db.modelGraph import ModelGraph
import inspect

__author__ = 'Brian Cherinka'


class MarvinDB(object):
    ''' Class designed to handle database related things with Marvin '''

    def __init__(self, *args, **kwargs):
        self.dbtype = kwargs.get('dbtype', None)
        self.db = None
        self.log = kwargs.get('log', None)
        self.error = []
        if self.dbtype:
            self._setupDB()
        if self.db:
            self._importModels()
        else:
            self.datadb = None
            self.sampledb = None
            self.dapdb = None
        self._setSession()
        self.testDbConnection()
        self._setModelGraph()

    def _setupDB(self):
        ''' Try to import the database '''
        try:
            from marvin.db.database import db
        except RuntimeError as e:
            print('RuntimeError raised: Problem importing db: {0}'.format(e))
            self.db = None
        else:
            self.db = db

    def _importModels(self):
        ''' Try to import the sql alchemy model classes '''

        try:
            import sdss.internal.database.utah.mangadb.SampleModelClasses as sampledb
        except Exception as e:
            print('Exception raised: Problem importing mangadb SampleModelClasses: {0}'.format(e))
            self.sampledb = None
        else:
            self.sampledb = sampledb

        try:
            import sdss.internal.database.utah.mangadb.DataModelClasses as datadb
        except Exception as e:
            print('Exception raised: Problem importing mangadb DataModelClasses: {0}'.format(e))
            self.datadb = None
        else:
            self.datadb = datadb

        try:
            import sdss.internal.database.utah.mangadb.DapModelClasses as dapdb
        except Exception as e:
            print('Exception raised: Problem importing mangadb DapModelClasses: {0}'.format(e))
            self.dapdb = None
        else:
            self.dapdb = dapdb

    def _setSession(self):
        ''' Sets the database session '''
        self.session = self.db.Session() if self.db else None

    def testDbConnection(self):
        ''' Test the database connection to ensure it works.  Sets a boolean variable isdbconnected '''
        if self.db and self.datadb:
            try:
                tmp = self.session.query(self.datadb.PipelineVersion).first()
            except Exception as e:
                self.isdbconnected = False
                self.error.append('Error connecting to manga database: {0}'.format(str(e)))
            else:
                self.isdbconnected = True
        else:
            self.isdbconnected = False

    def forceDbOff(self):
        ''' Force the database to turn off '''
        self.db = None
        self.session = None
        self.isdbconnected = False

    def generateClassDict(self, module=None, lower=None):
        ''' Generates a dictionary of the Model Classes, based on class name as key, to the object class.
            Selects only those classes in the module with attribute __tablename__
            lower = True makes class name key all lowercase
        '''
        if not module:
            module = self.datadb

        classdict = {}
        for model in inspect.getmembers(module, inspect.isclass):
            keyname = model[0].lower() if lower else model[0]
            if hasattr(model[1], '__tablename__'):
                classdict[keyname] = model[1]
        return classdict

    def buildUberClassDict(self):
        ''' Builds an uber class dictionary from all modelclasses '''
        classdict = {}
        models = [self.datadb, self.sampledb, self.dapdb]
        for model in models:
            if model:
                modelclasses = self.generateClassDict(module=model)
                classdict.update(modelclasses)
        return classdict

    def _setModelGraph(self):
        ''' Initiates the ModelGraph using all available ModelClasses '''
        models = list(filter(None, [self.datadb, self.sampledb, self.dapdb]))
        if models:
            self.modelgraph = ModelGraph(models)
        else:
            self.modelgraph = None
