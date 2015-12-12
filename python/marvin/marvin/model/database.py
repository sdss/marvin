#!/usr/bin/python

''' This file handles a database connection. It can simply be deleted if not needed.

	The example given is for a PostgreSQL database, but can be modified for any other.
'''
from __future__ import print_function
from __future__ import division

from sdss.internal.database.DatabaseConnection import DatabaseConnection
from flask import current_app as app
from pgpasslib import getpass

# default values here
db_info = {"port" : 5432}

# There is only one connection, but if more are possible
# one could use "if" statements based on a passed in parameter here.
# See the SDSSAPI product for an example.
try:
    db_info["host"] = app.config["DB_HOST"]
    db_info["database"] = app.config["DB_DATABASE"]
    db_info["user"] = app.config["DB_USER"]
    db_info["port"] = app.config["DB_PORT"]
    db_info["password"] = getpass(db_info["host"], db_info["port"], db_info["database"], db_info["user"])
except KeyError:
    current_app.logger.debug("ERROR: an expected key in the server configuration "
    "file was not found.")
    
# this format is only usable with PostgreSQL 9.2+
#dsn = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**db_info)
#database_connection_string = 'postgresql+psycopg2://%s:%s@%s:%s/%s' % (db_info["user"], db_info["password"], db_info["host"], db_info["port"], db_info["database"])

if db_info["host"]=='localhost':
    database_connection_string = 'postgresql+psycopg2:///%(database)s' % db_info
else:
    database_connection_string = 'postgresql+psycopg2://%(user)s:%(password)s@%(host)s:%(port)i/%(database)s' % db_info

try:
    db = DatabaseConnection()
except AssertionError:
    db = DatabaseConnection(database_connection_string=database_connection_string)
    engine = db.engine
    metadata = db.metadata
    Session = db.Session
    Base = db.Base
except KeyError as e:
    print("Necessary configuration value not defined.")
    raise e


# try:
#     from sdss.internal.database.connections.UtahMangaDatabaseConnection import db
# except AssertionError as e:
#     print("Something went wrong")
#     raise e
# except KeyError as e:
#     print("Necessary configuration value not defined.")
#     raise e
# else:
#     engine = db.engine
#     metadata = db.metadata
#     Session = db.Session
#     Base = db.Base
#     

