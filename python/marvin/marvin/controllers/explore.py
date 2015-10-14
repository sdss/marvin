#!/usr/bin/python

import os, glob, random

import flask
from flask import request, render_template, send_from_directory, current_app, session as current_session, jsonify
from manga_utils import generalUtils as gu
from collections import OrderedDict

from ..model.database import db
from ..utilities import setGlobalSession

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

from flask.ext.classy import FlaskView, route

try:
    from . import valueFromRequest
except ValueError:
    pass 


"""
class Explore(FlaskView):

    @route('/')
    def explore(self):
        explore={}
        explore['title'] = "Marvin | Explore"
        return render_template('explore.html', **explore)

    def get(self):
        pass

    def post(self):
        pass

explore_page = flask.Blueprint("explore_page", __name__)
Explore.register(explore_page)
"""

explore_page = flask.Blueprint("explore_page", __name__)

@explore_page.route('/explore/', methods=['GET'])
def explore():
    '''explore dataset page'''
    
    session = db.Session() 
    explore = {}
    explore['title'] = "Marvin | Explore"
    
    # set global session
    setGlobalSession() 
    
    return render_template("explore.html", **explore)
