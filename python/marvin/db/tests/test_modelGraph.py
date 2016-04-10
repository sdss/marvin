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
from marvin.db.modelGraph import ModelGraph, nx
from marvin import marvindb
from marvin.tools.tests import MarvinTest
from unittest import skipIf


@skipIf(not marvindb.session, 'no DB is available')
class TestModelGraph(MarvinTest):

    @classmethod
    def setupClass(cls):
        """Setups the test class."""

        cls.modelGraph = ModelGraph([marvindb.datadb, marvindb.sampledb])

    def testInitialitation(self):
        """Basic initialisation test."""

        # Initialises ModelGraph with datadb and sampledb
        modelGraphAll = ModelGraph([marvindb.datadb, marvindb.sampledb])

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

        self._testGetJoins(marvindb.datadb.Cube, [marvindb.datadb.Cube])
        self._testGetJoins(marvindb.datadb.Cube, ['mangadatadb.cube'],
                           format_out='tables')
        self._testGetJoins('mangadatadb.cube', [marvindb.datadb.Cube])

    def testGetJoins_oneModel_nexus(self):
        """Tests getJoins() with a single input model and a nexus."""

        self._testGetJoins(marvindb.datadb.Cube, [], nexus=marvindb.datadb.Cube)
        self._testGetJoins(marvindb.datadb.IFUDesign, [marvindb.datadb.IFUDesign],
                           nexus=marvindb.datadb.Cube)
        self._testGetJoins(marvindb.datadb.Fibers, [marvindb.datadb.IFUDesign, marvindb.datadb.Fibers],
                           nexus=marvindb.datadb.Cube)

    def testGetJoins_noPath(self):
        """Tests getJoins() when there is no path between two tables."""

        with self.assertRaises(nx.NetworkXNoPath):
            self._testGetJoins([marvindb.datadb.Cube, marvindb.datadb.MaskBit], [])

    def testGetJoin_fails(self):
        """Tests getJoins when input are wrong."""

        with self.assertRaises(ValueError):
            self._testGetJoins([1234], [])

        with self.assertRaises(ValueError):
            self._testGetJoins([], [])

        with self.assertRaises(AssertionError):
            self._testGetJoins([marvindb.datadb.Cube], [], format_out='bad_value')

        with self.assertRaises(AssertionError) as cm:
            self._testGetJoins([marvindb.datadb.Cube], [], nexus='bad_nexus')
            self.assertEqual(
                str(cm), 'nexus bad_nexus is not a node in the model graph')

    def testGetJoins_Models(self):
        """Tests getJoins() using a list of model classes."""

        models = [marvindb.datadb.Cube, marvindb.datadb.Fibers]
        expected = [marvindb.datadb.Cube, marvindb.datadb.IFUDesign, marvindb.datadb.Fibers]
        self._testGetJoins(models, expected)

        models = [marvindb.datadb.Cube, marvindb.datadb.Fibers, marvindb.datadb.FitsHeaderKeyword]
        expected = [marvindb.datadb.Cube, marvindb.datadb.IFUDesign, marvindb.datadb.Fibers,
                    marvindb.datadb.FitsHeaderValue, marvindb.datadb.FitsHeaderKeyword]
        self._testGetJoins(models, expected)

        models = [marvindb.datadb.Fibers, marvindb.sampledb.MangaTarget]
        expected = ['mangadatadb.fibers', 'mangadatadb.ifudesign',
                    'mangadatadb.cube', 'mangasampledb.manga_target']
        self._testGetJoins(models, expected, format_out='tables')

        models = [marvindb.datadb.Cube, marvindb.datadb.Fibers, marvindb.datadb.Cube]
        expected = [u'mangadatadb.cube', u'mangadatadb.ifudesign',
                    u'mangadatadb.fibers']
        self._testGetJoins(models, expected, format_out='tables')

    def testGetJoins_Models_nexus(self):
        """Tests getJoins() using a list of model classes and a nexus."""

        nexus = marvindb.datadb.Cube
        models = [marvindb.datadb.Cube, marvindb.datadb.Fibers, marvindb.datadb.FitsHeaderKeyword]
        expected = [marvindb.datadb.IFUDesign, marvindb.datadb.Fibers, marvindb.datadb.FitsHeaderValue,
                    marvindb.datadb.FitsHeaderKeyword]
        expected_tables = ['mangadatadb.ifudesign', 'mangadatadb.fibers',
                           'mangadatadb.fits_header_value',
                           'mangadatadb.fits_header_keyword']

        self._testGetJoins(models, expected, nexus=nexus)
        self._testGetJoins(models, expected_tables, format_out='tables',
                           nexus=nexus)
