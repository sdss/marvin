#!/usr/bin/python

import os, glob, random

import flask
from flask import request, render_template, send_from_directory, current_app, session as current_session, jsonify
from manga_utils import generalUtils as gu
from collections import OrderedDict

from ..model.database import db
from ..utilities import setGlobalVersion

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try:
    from . import valueFromRequest
except ValueError:
    pass
    
    
explore_page = flask.Blueprint("explore_page", __name__)

@explore_page.route('/explore.html', methods=['GET'])
@explore_page.route('/marvin/explore.html', methods=['GET'])
def explore():
    ''' explore dataset page '''
    
    session = db.Session() 
    explore = {}
    explore['title'] = "Marvin | Explore"
    
    # set global version    
    try: version = current_session['currentver']
    except: 
        setGlobalVersion()
        version = current_session['currentver']
        
    
    return render_template("explore.html", **explore)
    