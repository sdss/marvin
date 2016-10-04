#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-04-26 09:20:35
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-04-26 09:20:35 by Brian Cherinka
    Last Modified On: 2016-04-26 09:20:35 by Brian
'''

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.event import listen
from sqlalchemy.pool import Pool


def clearSearchPathCallback(dbapi_con, connection_record):
    '''
    When creating relationships across schema, SQLAlchemy
    has problems when you explicitly declare the schema in
    ModelClasses and it is found in search_path.

    The solution is to set the search_path to "$user" for
    the life of any connection to the database. Since there
    is no (or shouldn't be!) schema with the same name
    as the user, this effectively makes it blank.

    This callback function is called for every database connection.

    For the full details of this issue, see:
    http://groups.google.com/group/sqlalchemy/browse_thread/thread/88b5cc5c12246220

    dbapi_con - type: psycopg2._psycopg.connection
    connection_record - type: sqlalchemy.pool._ConnectionRecord
    '''
    cursor = dbapi_con.cursor()
    cursor.execute('SET search_path TO "$user",functions')
    dbapi_con.commit()

listen(Pool, 'connect', clearSearchPathCallback)


class DatabaseConnection(object):
    '''This class defines an object that makes a connection to a database.
       The "DatabaseConnection" object takes as its parameter the SQLAlchemy
       database connection string.

       This class is best called from another class that contains the
       actual connection information (so that it can be reused for different
       connections).

       This class implements the singleton design pattern. The first time the
       object is created, it *requires* a valid database connection string.
       Every time it is called via:

       db = DatabaseConnection()

       the same object is returned and contains the connection information.
    '''
    _singletons = dict()

    def __new__(cls, database_connection_string=None, expire_on_commit=True):
        """This overrides the object's usual creation mechanism."""

        if cls not in cls._singletons:
            assert database_connection_string is not None, "A database connection string must be specified!"
            cls._singletons[cls] = object.__new__(cls)

            # ------------------------------------------------
            # This is the custom initialization
            # ------------------------------------------------
            me = cls._singletons[cls]  # just for convenience (think "self")

            me.database_connection_string = database_connection_string

            # change 'echo' to print each SQL query (for debugging/optimizing/the curious)
            me.engine = create_engine(me.database_connection_string, echo=False)

            me.metadata = MetaData()
            me.metadata.bind = me.engine
            me.Base = declarative_base(bind=me.engine)
            me.Session = scoped_session(sessionmaker(bind=me.engine, autocommit=True,
                                                     expire_on_commit=expire_on_commit))
            # ------------------------------------------------

        return cls._singletons[cls]


