#!/usr/bin/python

import os, glob, random

import flask
from flask import request, render_template, send_from_directory, current_app, session as current_session, jsonify
from manga_utils import generalUtils as gu
from collections import OrderedDict

from ..model.database import db
from ..utilities import setGlobalVersion

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try: from inspection.marvin import Inspection
except: from marvin.inspection import Inspection

try:
    from . import valueFromRequest
except ValueError:
    pass
    
  
test_page = flask.Blueprint("test_page", __name__)

@test_page.route('/tests/', methods=['GET'])
@test_page.route('/marvin/tests/', methods=['GET'])
def test():
    '''unit test page'''
    
    # set global version
    try: 
        version = current_session['currentver']
        dapversion = current_session['currentdapver']
    except: 
        setGlobalVersion()
        version = current_session['currentver']
        dapversion = current_session['currentdapver']

    session = db.Session() 
    tests = {}
    tests['title'] = "Marvin | Testing"
    tests['inspection'] = inspection = Inspection(current_session)
    tests['cube'] = {'ifu':{'name':'9101'},'plate':'7443'}
    tests['version'] = version
    tests['dapversion'] = dapversion

    return render_template("tests.html", **tests)



    