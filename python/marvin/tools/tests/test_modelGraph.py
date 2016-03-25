#!/usr/bin/env python
# encoding: utf-8
"""

test_modelGraph.py

Created by José Sánchez-Gallego on 24 Mar 2016.
Licensed under a 3-clause BSD license.

Revision history:
    24 Mar 2016 J. Sánchez-Gallego
      Initial version

"""

from __future__ import division
from __future__ import print_function
from marvin.tools.query.modelGraph import ModelGraph, nx
from marvin import sampledb, datadb, session
from marvin.tools.tests import MarvinTest
from unittest import skipIf


@skipIf(not session, 'no DB is available')
class TestModelGraph(MarvinTest):

    @classmethod
    def setupClass(cls):
        """Setups the test class."""

        cls.modelGraph = ModelGraph([datadb, sampledb])

    def testInitialitation(self):
        """Basic initialisation test."""

        # Initialises ModelGraph without arguments.
        modelGraphData = ModelGraph()
        self.assertGreater(len(modelGraphData.nodes), 0)
        self.assertGreater(len(modelGraphData.edges), 0)

        # Initialises ModelGraph with datadb and sampledb
        modelGraphAll = ModelGraph([datadb, sampledb])
        self.assertGreater(len(modelGraphAll.nodes), len(modelGraphData.nodes))
        self.assertGreater(len(modelGraphAll.edges), len(modelGraphData.edges))

        testTables = ['mangasampledb.manga_target', 'mangasampledb.character',
                      'mangadatadb.cube', 'mangadatadb.fibers']
        for table in testTables:
            self.assertIn(table, modelGraphAll.nodes)

    def _testGetJoins(self, models, expected, nexus=None,
                      format_out='models'):
        """Convenience method to test getJoins()."""

        joins = self.modelGraph.getJoins(models, format_out=format_out,
                                         nexus=nexus)

        self.assertListEqual(joins, expected)

    def testGetJoins_oneModel(self):
        """Tests getJoins() with a single input model."""

        self._testGetJoins(datadb.Cube, [datadb.Cube])
        self._testGetJoins(datadb.Cube, ['mangadatadb.cube'],
                           format_out='tables')
        self._testGetJoins('mangadatadb.cube', [datadb.Cube])

    def testGetJoins_oneModel_nexus(self):
        """Tests getJoins() with a single input model and a nexus."""

        self._testGetJoins(datadb.Cube, [], nexus=datadb.Cube)
        self._testGetJoins(datadb.IFUDesign, [datadb.IFUDesign],
                           nexus=datadb.Cube)
        self._testGetJoins(datadb.Fibers, [datadb.IFUDesign, datadb.Fibers],
                           nexus=datadb.Cube)

    def testGetJoins_noPath(self):
        """Tests getJoins() when there is no path between two tables."""

        with self.assertRaises(nx.NetworkXNoPath):
            self._testGetJoins([datadb.Cube, datadb.MaskBit], [])

    def testGetJoin_fails(self):
        """Tests getJoins when input are wrong."""

        with self.assertRaises(ValueError):
            self._testGetJoins([1234], [])

        with self.assertRaises(ValueError):
            self._testGetJoins([], [])

        with self.assertRaises(AssertionError):
            self._testGetJoins([datadb.Cube], [], format_out='bad_value')

        with self.assertRaises(AssertionError) as cm:
            self._testGetJoins([datadb.Cube], [], nexus='bad_nexus')
            self.assertEqual(
                str(cm), 'nexus bad_nexus is not a node in the model graph')

    def testGetJoins_Models(self):
        """Tests getJoins() using a list of model classes."""

        models = [datadb.Cube, datadb.Fibers]
        expected = [datadb.Cube, datadb.IFUDesign, datadb.Fibers]
        self._testGetJoins(models, expected)

        models = [datadb.Cube, datadb.Fibers, datadb.FitsHeaderKeyword]
        expected = [datadb.Cube, datadb.IFUDesign, datadb.Fibers,
                    datadb.FitsHeaderValue, datadb.FitsHeaderKeyword]
        self._testGetJoins(models, expected)

        models = [datadb.Fibers, sampledb.MangaTarget]
        expected = ['mangadatadb.fibers', 'mangadatadb.ifudesign',
                    'mangadatadb.cube', 'mangasampledb.manga_target']
        self._testGetJoins(models, expected, format_out='tables')

        models = [datadb.Cube, datadb.Fibers, datadb.Cube]
        expected = [u'mangadatadb.cube', u'mangadatadb.ifudesign',
                    u'mangadatadb.fibers']
        self._testGetJoins(models, expected, format_out='tables')

    def testGetJoins_Models_nexus(self):
        """Tests getJoins() using a list of model classes and a nexus."""

        nexus = datadb.Cube
        models = [datadb.Cube, datadb.Fibers, datadb.FitsHeaderKeyword]
        expected = [datadb.IFUDesign, datadb.Fibers, datadb.FitsHeaderValue,
                    datadb.FitsHeaderKeyword]
        expected_tables = ['mangadatadb.ifudesign', 'mangadatadb.fibers',
                           'mangadatadb.fits_header_value',
                           'mangadatadb.fits_header_keyword']

        self._testGetJoins(models, expected, nexus=nexus)
        self._testGetJoins(models, expected_tables, format_out='tables',
                           nexus=nexus)
