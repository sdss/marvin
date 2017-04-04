#!/usr/bin/env python2
# encoding: utf-8
#
# test_db_switch.py
#
# Created by José Sánchez-Gallego on Sep 7, 2016.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


def create_connection(db_name):
    """Creates the connection and import the model classes."""

    from marvin.db.DatabaseConnection import DatabaseConnection

    database_connection_string = 'postgresql+psycopg2:///{0}'.format(db_name)
    db = DatabaseConnection(database_connection_string=database_connection_string)

    import marvin.db.models.DataModelClasses as mangaData

    return db, mangaData


def perform_query(db, mangaData):
    """Performs a simple query and return the value."""

    session = db.Session()

    xfocal = session.query(mangaData.Cube.xfocal).filter(
        mangaData.Cube.plate == 8485, mangaData.Cube.mangaid == '1-209232').join(
            mangaData.PipelineInfo, mangaData.PipelineVersion).filter(
                mangaData.PipelineVersion.version == 'v1_5_1').one()

    return xfocal


# db_name = 'manga'
# db, mangaData = create_connection(db_name)
# print(perform_query(db, mangaData))

# db_name_copy = 'manga_copy'
# db, mangaData = create_connection(db_name_copy)
# print(perform_query(db, mangaData))
