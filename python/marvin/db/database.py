#!/usr/bin/python

''' This file handles a database connection. It can simply be deleted if not needed.

    The example given is for a PostgreSQL database, but can be modified for any other.
'''
from __future__ import division, print_function

import os

import yaml
from marvin import config
from marvin.db.DatabaseConnection import DatabaseConnection
from pgpasslib import getpass


# Read in the db connection configuration
dbconfigfile = 'dbconfig.ini'
dbconfigfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), dbconfigfile)

try:
    with open(dbconfigfile, 'r') as ff:
        rawfile = ff.read()

except IOError as e:
    raise RuntimeError('IOError: Could not open dbconfigfile {0}:{1}'.format(dbconfigfile, e))
dbdict = yaml.load(rawfile)

# select the appropriate configuration from config
if config.db:
    db_info = dbdict[config.db]
    try:
        if 'password' not in db_info:
            db_info['password'] = getpass(db_info['host'], db_info['port'], db_info['database'], db_info['user'])
    except KeyError:
        raise RuntimeError('ERROR: invalid server configuration')
else:
    raise RuntimeError('Error: could not determine db to connect to: {0}'.format(config.db))

# this format is only usable with PostgreSQL 9.2+
# dsn = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**db_info)
# database_connection_string = 'postgresql+psycopg2://%s:%s@%s:%s/%s' % (db_info["user"], db_info["password"], db_info["host"], db_info["port"], db_info["database"])

# Build the database connection string
if db_info["host"] == 'localhost':
    database_connection_string = 'postgresql+psycopg2:///%(database)s' % db_info
else:
    database_connection_string = 'postgresql+psycopg2://%(user)s:%(password)s@%(host)s:%(port)i/%(database)s' % db_info

# Make a database connection
try:
    db = DatabaseConnection()
except AssertionError as e:
    db = DatabaseConnection(database_connection_string=database_connection_string)
    engine = db.engine
    metadata = db.metadata
    Session = db.Session
    Base = db.Base
except KeyError as e:
    print("Necessary configuration value not defined.")
    raise RuntimeError('KeyError: Necessary configuration value not defined: {0}'.format(e))
