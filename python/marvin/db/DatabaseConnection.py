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
from marvin.core import caching_query
from marvin import config
from hashlib import md5
from dogpile.cache.region import make_region
import os

# DOGPILE CACHING SETUP

# dogpile cache regions.  A home base for cache configurations.
regions = {}

dogpath = os.environ.get('MANGA_SCRATCH_DIR', None)
if dogpath and os.path.isdir(dogpath):
    dogpath = dogpath
else:
    dogpath = os.path.expanduser('~')

dogroot = os.path.join(dogpath, 'dogpile_data')
if not os.path.isdir(dogroot):
    os.makedirs(dogroot)


# make an nsa region
def make_nsa_region(name):
    reg = make_region(key_mangler=md5_key_mangler).configure(
            'dogpile.cache.dbm',
            expiration_time=3600,
            arguments={'filename': os.path.join(dogroot, '{0}_cache.dbm'.format(name))}
        )
    return reg


# db hash key
def md5_key_mangler(key):
    """Receive cache keys as long concatenated strings;
    distill them into an md5 hash.

    """
    return md5(key.encode('ascii')).hexdigest()

# configure the "default" cache region.
regions['default'] = make_region(
            # the "dbm" backend needs string-encoded keys
            key_mangler=md5_key_mangler
        ).configure(
        # using type 'file' to illustrate
        # serialized persistence.  Normally
        # memcached or similar is a better choice
        # for caching.
        # 'dogpile.cache.dbm',  # file-based backend
        'dogpile.cache.memcached',  # memcached-based backend
        expiration_time=3600,
        arguments={
            'url': "127.0.0.1:11211"  # memcached option
            # "filename": os.path.join(dogroot, "cache.dbm") # file option
        }
    )

for mpl in config._allowed_releases.keys():
    nsacache = 'nsa_{0}'.format(mpl.lower().replace('-', ''))
    regions[nsacache] = make_nsa_region(nsacache)

#regions['nsa_mpl5'] = make_nsa_region('nsa_mpl5')
#regions['nsa_mpl4'] = make_nsa_region('nsa_mpl4')
#regions['nsa_mpl6'] = make_nsa_region('nsa_mpl6')


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
    cursor.execute('SET search_path TO "$user",functions,public')
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
            me.engine = create_engine(me.database_connection_string, echo=False, pool_size=10, pool_recycle=1800)

            me.metadata = MetaData()
            me.metadata.bind = me.engine
            me.Base = declarative_base(bind=me.engine)
            me.Session = scoped_session(sessionmaker(bind=me.engine, autocommit=True,
                                                     query_cls=caching_query.query_callable(regions),
                                                     expire_on_commit=expire_on_commit))
            # ------------------------------------------------

        return cls._singletons[cls]


