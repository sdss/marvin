#!/usr/bin/python

import os, glob, random

import flask
from flask import request, render_template, send_from_directory, current_app, session as current_session, jsonify
from manga_utils import generalUtils as gu
from collections import OrderedDict

from ..model.database import db
from ..utilities import setGlobalSession

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
    
    # set global session variables
    setGlobalSession()

    session = db.Session() 
    tests = {}
    tests['title'] = "Marvin | Testing"
    tests['inspection'] = inspection = Inspection(current_session)
    tests['cube'] = {'ifu':{'name':'9101'},'plate':'7443'}
    tests['version'] = current_session['currentver']
    tests['dapversion'] = current_session['currentdapver']

    return render_template("tests.html", **tests)



    