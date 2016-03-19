#!/usr/bin/env python
# encoding: utf-8
"""

modelPaths.py

Created by José Sánchez-Gallego on 18 Mar 2016.
Licensed under a 3-clause BSD license.

Revision history:
    18 Mar 2016 J. Sánchez-Gallego
      Initial version

"""

from __future__ import division
from __future__ import print_function
from sqlalchemy.ext.declarative import DeclarativeMeta
import itertools
from marvin import datadb
import numpy as np
import networkx as nx


def getModels(module):
    """Returns a list with all the ModelClasses for a module.

    E.g., for `datadb` it returns PipelineCompletionStatus, PipelineInfo,
    IFUDesign, Cube, Sample, etc.

    """

    models = []
    attrs = vars(module).keys()

    # Loops over all the attributes in the module and selects those that are
    # model classes.
    for attr in attrs:
        testModel = getattr(module, attr)
        if (isinstance(testModel, DeclarativeMeta) and
                hasattr(testModel, '__table__') and testModel.__table__.name):
            models.append(getattr(module, attr))

    return models


class ModelGraph(object):

    def __init__(self, modelSchemas=datadb):
        """Creates a `networkx` graph between the model classes.

        This class creates a graph in which each table in the input
        `modelSchemas` is a node, and the foreign keys relating them act as
        edges (links). The class includes some convenience methods to find the
        shortest path between two or more tables/nodes.

        The main purpose of the class is to help finding the necessary tables
        to join for a SQLalchemy query to work. For instance, one may want
        to perform a query such as `'ifu.nfiber == 127 and nsa.nsa_z > 0.1'`.
        From that query only two model classes are retrieved:
        `mangadatadb.IFUDesign` and `mangasampledb.Nsa`, which don't have a
        direct relationship. However, it is possible to perform this query if
        we add the following models to the join: `mangadatadb.Cube`,
        `mangasampledb.Target`. Creating a graph with all the nodes and
        relationships in the database makes finding the shortest path between
        two tables trivial.

        Parameters
        ----------
        modelSchemas : module or list of modules
            A module or list of modules containing model classes. Each model
            class will be considered a node in the graph, and their
            relationships will define the edges between nodes.

        Example
        -------
        An example of use ::
          >> graph = ModelGraph([datadb, sampledb])
          >> graph.join_models([IFUDesign, Nsa])
          >> [IFUDesign, Cube, Target, Nsa]

        """

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

    def getJoins(self, models, format_in='tables', format_out='tables'):
        """Returns a list all model classes needed to perform a join.

        Given a list of `models`, finds the shortest join paths between any two
        elements in the list. Returns a list of all the tables or models that
        need to be joined to perform a query on `models`.

        Parameters
        ----------
        models : list of model classes or tablenames
            A list of the tables or model classes to join. If tablenames,
            the full path, `schema.table` must be provided.
        format_in, format_out : string
            `format_in` defines the type of input in `models`, and can be
            either `'models'` for model classes, or `'tables'` for tablenames.
            `format_out`, which accepts the same values, determines the type
            of the elements in the returned list.

        Returns
        -------
        join_list : list
            A list of all the model classes or tablenames (depending on the
            value of `format_out`) needed to connect all the elements in
            `models`. The original elements in `models` are also included.

        """

        models = np.atleast_1d(models)
        format_in = format_in.lower()
        format_out = format_out.lower()

        assert format_in in ['models', 'tables'], \
            'format_in must be either \'models\' or \'tables\'.'

        assert format_out in ['models', 'tables'], \
            'format_out must be either \'models\' or \'tables\'.'

        if format_in == 'models':
            tables = [self.getTablePath(model) for model in models]
        else:
            tables = models

        for table in tables:
            assert table in self.graph.nodes(), \
                'table {0} is not a node in the model graph.'.format(table)

        if len(models) == 0:
            raise ValueError('input list of models/tables to join is empty.')

        elif len(models) == 1:
            # Simple case in which we only have one table to join. We just
            # return the same table / model, depending on format_out
            if format_out == 'tables':
                return tables
            else:
                return [self.graph.node[tables[0]]['model']]

        else:
            # We get all possible combinations of two elements in the input
            # list of models, and find the shortest path between them.
            joins = []
            for tableA, tableB in itertools.combinations(tables, r=2):
                try:
                    path = nx.shortest_path(self.graph, tableA, tableB)
                except nx.NetworkXNoPath:
                    raise nx.NetworkXNoPath(
                        'it is not possible to join tables {0} and {1}. '
                        'Please, review your query.'.format(tableA, tableB))

                for table in path:
                    if format_out == 'tables':
                        newJoin = table
                    else:
                        newJoin = self.graph.node[table]['model']

                    if newJoin not in joins:
                        joins.append(newJoin)

            return joins
