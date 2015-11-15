#!/usr/bin/python

import os

import flask
from flask import request, render_template, send_from_directory, current_app, session as current_session, jsonify,abort

from ..model.database import db

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try:
    from . import valueFromRequest,processRequest
except ValueError:
    pass
    
    
doc_page = flask.Blueprint("doc_page", __name__)

@doc_page.route('/help/', methods=['POST'])
def help():
    ''' Help page '''

    helpdict = {}

    return render_template('help.html')

@doc_page.route('/documentation/', methods=['GET'])
def doc():
    ''' Documentation here. '''
    
    session = db.Session() 
    doc = {}   

    try: return render_template("documentation.html", **doc)
    except: abort(404) 

@doc_page.errorhandler(404)
def page_not_found(e):
    error={}
    error['page']='Documentation'
    error['title']='Marvin | Page Not Found'
    return render_template('errors/page_not_found.html',**error) 


