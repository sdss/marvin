#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-03-28 23:30:14
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-03-28 23:30:14 by Brian Cherinka
    Last Modified On: 2016-03-28 23:30:14 by Brian

'''
from __future__ import print_function, division
from brain.db.modelGraph import ModelGraph
import inspect

__author__ = 'Brian Cherinka'


class MarvinDB(object):
    ''' Class designed to handle database related things with Marvin '''

    def __init__(self, dbtype=None, log=None, allowed_releases=None):
        self.dbtype = dbtype
        self.db = None
        self.log = log
        self.allowed_releases = allowed_releases
        self.error = []
        self.spaxelpropdict = None
        self.datadb = None
        self.dapdb = None
        self.sampledb = None
        self._init_the_db()

    def _init_the_db(self):
        ''' Initialize the db '''
        if self.dbtype:
            self._setupDB()
        if self.db:
            self._importModels()
        self._setSession()
        self.testDbConnection()
        self._setModelGraph()
        self.cache_bits = []
        if self.db:
           self._addCache()

    def _setupDB(self):
        ''' Try to import the database '''
        # time - 14.8 ms
        try:
            from marvin.db.database import db
        except RuntimeError as e:
            self.log.debug('RuntimeError raised: Problem importing db: {0}'.format(e))
            self.db = None
        except ImportError as e:
            self.log.debug('ImportError raised: Problem importing db: {0}'.format(e))
            self.db = None
        else:
            self.db = db

    def _importModels(self):
        ''' Try to import the sql alchemy model classes '''

        # tested lazy imports - speeds init until they get called
        # import lazy_import
        # sampledb = lazy_import.lazy_module("marvin.db.models.SampleModelClasses")

        # time 1.6 seconds
        try:
            import marvin.db.models.SampleModelClasses as sampledb
        except Exception as e:
            self.log.debug('Exception raised: Problem importing mangadb SampleModelClasses: {0}'.format(e))
        else:
            self.sampledb = sampledb

        try:
            import marvin.db.models.DataModelClasses as datadb
        except Exception as e:
            self.log.debug('Exception raised: Problem importing mangadb DataModelClasses: {0}'.format(e))
        else:
            self.datadb = datadb

        try:
            import marvin.db.models.DapModelClasses as dapdb
        except Exception as e:
            self.log.debug('Exception raised: Problem importing mangadb DapModelClasses: {0}'.format(e))
        else:
            self.dapdb = dapdb
            self.spaxelpropdict = self._setSpaxelPropDict()

    def has_models(self):
        ''' check if the marvin db has all the models properly loaded '''
        isdata = self.datadb is not None
        isdap = self.dapdb is not None
        issample = self.sampledb is not None
        self.log.info('datadb? {0}'.format(isdata))
        self.log.info('dapdb? {0}'.format(isdap))
        self.log.info('sampledb? {0}'.format(issample))
        return all([isdata, isdap, issample])

    def _setSpaxelPropDict(self):
        ''' Set the SpaxelProp lookup dictionary '''

        # time - 38 us
        from marvin.utils.datamodel.dap import datamodel
        spdict = {}
        for release in self.allowed_releases:
            if release in datamodel:
                dm = datamodel[release]
                spdict.update({release: dm.property_table})
        return spdict

    def _getSpaxelProp(self):
        ''' Get the correct SpaxelProp class given an release '''

        inspdict = self._release in self.spaxelpropdict
        if inspdict:
            specific_spaxelprop = {'full': self.spaxelpropdict[self._release], 'clean':
                                   'Clean{0}'.format(self.spaxelpropdict[self._release])}
        else:
            specific_spaxelprop = {'full': None, 'clean': None}
        return specific_spaxelprop

    def _setSession(self):
        ''' Sets the database session '''
        self.session = self.db.Session() if self.db else None

    def testDbConnection(self):
        ''' Test the database connection to ensure it works.  Sets a boolean variable isdbconnected '''

        # time - 4.7 ms
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
        self.datadb = None
        self.dapdb = None
        self.sampledb = None

    def forceDbOn(self, dbtype=None):
        ''' Force the database to turn on '''
        self._init_the_db()

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
                # only include the spaxelprop table matching the MPL version
                if 'SpaxelProp' in keyname:
                    if keyname in self._getSpaxelProp().values():
                        classdict[keyname] = model[1]
                else:
                    classdict[keyname] = model[1]
        return classdict

    def buildUberClassDict(self, release=None):
        ''' Builds an uber class dictionary from all modelclasses '''
        self._release = release
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

    def _addCache(self):
        ''' Initialize dogpile caching for relationships

        Caching options.   A set of three RelationshipCache options
        which can be applied to Query(), causing the "lazy load"
        of these attributes to be loaded from cache.

        '''

        if self.datadb:
            self.cache_bits.append(self.datadb.data_cache)

        if self.sampledb:
            self.cache_bits.append(self.sampledb.sample_cache)

        if self.dapdb:
            self.cache_bits.append(self.dapdb.dap_cache)
