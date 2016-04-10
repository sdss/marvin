#!/usr/bin/env python
# encoding: utf-8
"""

modelPaths.py

Created by José Sánchez-Gallego on 18 Mar 2016.
Licensed under a 3-clause BSD license.

Revision history:
    18 Mar 2016 J. Sánchez-Gallego
      Initial version
    29 Mar 2016 - Removed default datadb dependency - B. Cherinka

"""

from __future__ import division
from __future__ import print_function
from sqlalchemy.ext.declarative import DeclarativeMeta
import itertools
import numpy as np
import networkx as nx


def isModel(model):
    """Return True if the input is a Model Class."""

    if (isinstance(model, DeclarativeMeta) and hasattr(model, '__table__') and
            model.__table__.name):
        return True
    else:
        False


def getModels(module):
    """Returns a list with all the ModelClasses for a module.

    E.g., for ``datadb`` it returns PipelineCompletionStatus, PipelineInfo,
    IFUDesign, Cube, Sample, etc.

    """

    models = []
    attrs = vars(module).keys()

    # Loops over all the attributes in the module and selects those that are
    # model classes.
    for attr in attrs:
        testModel = getattr(module, attr)
        if isModel(testModel):
            models.append(getattr(module, attr))

    return models


class ModelGraph(object):
    """Creates a `networkx` graph between the model classes.

    This class creates a graph in which each table in the input
    ``modelSchemas`` is a node, and the foreign keys relating them act as
    edges (links). The class includes some convenience methods to find the
    shortest path between two or more tables/nodes.

    The main purpose of the class is to help finding the necessary tables
    to join for a SQLalchemy query to work. For instance, one may want
    to perform a query such as ``'ifu.nfiber == 127 and nsa.nsa_z > 0.1'``.
    From that query only two model classes are retrieved:
    ``mangadatadb.IFUDesign`` and ``mangasampledb.Nsa``, which don't have a
    direct relationship. However, it is possible to perform this query if
    we add the following models to the join: ``mangadatadb.Cube``,
    ``mangasampledb.Target``. Creating a graph with all the nodes and
    relationships in the database makes finding the shortest path between
    two tables trivial.

    Parameters:
        modelSchemas (module or list of modules):
            A module or list of modules containing model classes.
            Each model class will be considered a node in the graph, and their
            relationships will define the edges between nodes.

    Example:
      >>> graph = ModelGraph([datadb, sampledb])
      >>> graph.join_models([IFUDesign, Nsa])
          [IFUDesign, Cube, Target, Nsa]

    """

    def __init__(self, modelSchemas):

        self.schemas = np.atleast_1d(modelSchemas)
        self.models = {}

        # Initialites a graph in which the tables (schema.tablename) will be
        # the nodes.
        self.graph = nx.Graph()

        # Creates the nodes
        for schema in self.schemas:
            self._createNodes(schema)

        # Creates the edges from the relationships between models.
        for schema in self.schemas:
            self._createEdges(schema)

    @property
    def nodes(self):
        """Shortcut to self.graph.nodes()."""
        return self.graph.nodes()

    @property
    def edges(self):
        """Shortcut to self.graph.edges()."""
        return self.graph.edges()

    @staticmethod
    def getTablePath(model):
        """From a model, returns the table path in the form schema.table."""

        tableName = model.__table__.name
        schemaName = model.__table__.schema
        fullPath = schemaName + '.' + tableName

        return fullPath

    def _createNodes(self, schema):
        """Creates the nodes for the tables in a schema."""

        # Gets the models in a schema module
        self.models[schema.__name__] = getModels(schema)

        # Adds the table name of each model as a node.
        for model in self.models[schema.__name__]:
            self.graph.add_node(self.getTablePath(model), model=model)

    def _createEdges(self, schema):
        """Creates the edges between nodes in the table graph."""

        # We loop over each model, get the foreign keys, and create an edge
        # between parent and child. Networkx will automatically ignore inverse
        # edges (i.e., if we add an edge between Cube and Wavelength, and later
        # add the inverse edge between Wavelength and Cube, the latter edge
        # will be ignored, as a previous link between the two nodes already
        # exists).
        for model in self.models[schema.__name__]:

            foreignKeys = list(model.__table__.foreign_keys)
            for fKey in foreignKeys:
                childSchemaName = fKey.column.table.schema
                childTableName = fKey.column.table.name
                childFullPath = childSchemaName + '.' + childTableName
                if childFullPath not in self.graph.nodes():
                    continue
                else:
                    parent = self.getTablePath(model)
                    self.graph.add_edge(parent, childFullPath)

    def getJoins(self, models, format_out='tables', nexus=None):
        """Returns a list all model classes needed to perform a join.

        Given a list of ``models``, finds the shortest join paths between any
        two elements in the list. Returns a list of all the tables or models
        that need to be joined to perform a query on ``models``.

        Parameters
        ----------
        models : list of model classes or tablenames
            A list of the tables or model classes to join. If tablenames,
            the full path, ``schema.table`` must be provided. The format of the
            input is determined automatically from the type of the first
            element.
        format_out : string
            Defines the type of elements in the returned list. If ``'models'``,
            the returned elements will be model classes, if ``'tables'`` they
            will be the corresponding table paths (schema.table).
        nexus : string, model class, or None
            If None, the method will find the paths between each combination of
            two elements in the input ``model`` list. If a table is provided
            (either as a table path or as a model class), the returned list
            will be the shorted path between `nexus` and each on of the tables
            in ``models``. The ``nexus`` table won't be included in the output.

        Returns
        -------
        join_list : list
            A list of all the model classes or tablenames (depending on the
            value of ``format_out``) needed to connect all the elements in
            ``models``. The original elements in `models` are also included.

        """

        models = np.atleast_1d(models)
        format_out = format_out.lower()

        assert format_out in ['models', 'tables'], \
            'format_out must be either \'models\' or \'tables\'.'

        if len(models) == 0:
            raise ValueError('input list of models/tables to join is empty.')

        # Removes duplicates in input list.
        seen = set()
        seen_add = seen.add
        models = [xx for xx in models if not (xx in seen or seen_add(xx))]

        # Determines the type of input
        if isModel(models[0]):
            format_in = 'models'
        elif isinstance(models[0], (str, np.unicode_)):
            format_in = 'tables'
        else:
            raise ValueError('the format of the input list '
                             'cannot be understood.')

        # We create a list of the table paths for each input. We'll use
        # this list internally, and later format the output depending on
        # format_out.
        if format_in == 'models':
            tables = [self.getTablePath(model) for model in models]
        else:
            tables = models

        if nexus and isModel(nexus):
            nexus = self.getTablePath(nexus)
            assert nexus in self.graph.nodes(), \
                'nexus {0} is not a node in the model graph.'.format(nexus)

        for table in tables:
            assert table in self.graph.nodes(), \
                'table {0} is not a node in the model graph.'.format(table)

        if len(models) == 1:
            # Simple case in which we only have one table to join. We just
            # return the same table / model, depending on format_out
            # If nexus is defined, we recursively call getJoins().

            if not nexus:
                if format_out == 'tables':
                    return tables
                else:
                    return [self.graph.node[tables[0]]['model']]
            else:
                path = self.getJoins(models=[nexus, tables[0]], nexus=None,
                                     format_out=format_out)
                # Removes the nexus
                return path[1:]

        else:
            # We get all possible combinations of two elements in the input
            # list of models, and find the shortest path between them.
            # If nexus is defined, we get the joins between nexus and each item
            # in tables.

            joins = []

            if not nexus:
                for tableA, tableB in itertools.combinations(tables, r=2):
                    newJoins = self._getShortestPath(tableA, tableB,
                                                     format_out=format_out)
                    self._joinList(joins, newJoins)

            else:
                for tableB in tables:
                    newJoins = self._getShortestPath(nexus, tableB,
                                                     format_out=format_out,
                                                     removeA=True)
                    self._joinList(joins, newJoins)

            return joins

    def _joinList(self, listA, listB):
        """Joins two lists excluding duplicates. Join is made in place."""

        for table in listB:
            if table not in listA:
                listA.append(table)

        return listA

    def _getShortestPath(self, tableA, tableB, format_out='tables',
                         removeA=False):
        """Gets the shortest path between two nodes."""

        try:
            path = nx.shortest_path(self.graph, tableA, tableB)
        except nx.NetworkXNoPath:
            raise nx.NetworkXNoPath(
                'it is not possible to join tables {0} and {1}. '
                'Please, review your query.'.format(tableA, tableB))

        if removeA:
            path.remove(tableA)

        if format_out == 'tables':
            pathSet = path
        else:
            pathSet = [self.graph.node[table]['model'] for table in path]

        return pathSet
