#!/usr/bin/python

import os, glob, random

import flask
from flask import request, render_template, send_from_directory, current_app, session as current_session, jsonify
from manga_utils import generalUtils as gu
from collections import OrderedDict

from ..model.database import db
from ..utilities import setGlobalSession, processTableData

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


@explore_page.route('/explore/searchresults/', methods=['GET','POST'])
def results():
    ''' explore search results '''

    explore = {}
    explore['title'] = "Marvin | Results"
    result = {'status':0,'message':None}
    table = valueFromRequest(key='hiddendata',request=request, default=None)
 
    print('explore request', request)

    print('table', table)

    if table != 'null' and table != None:
        try:
            newtable = processTableData(table) 
        except AttributeError as e:
            result['message'] = 'AttributeError getting rsync, in processTableData: {0}'.format(e)
            result['status'] = -1
            #return jsonify(result=result)
        except KeyError as e:
            result['message'] = 'KeyError getting rsync, in processTableData: {0}'.format(e)
            result['status'] = -1
            #return jsonify(result=result)
        print('result',result)
    else: newtable = None   

    return render_template("explore.html", **explore)

@explore_page.route('/explore/', methods=['GET'])
def explore():
    '''explore dataset page'''
    
    session = db.Session() 
    explore = {}
    explore['title'] = "Marvin | Explore"
    
    # set global session
    setGlobalSession() 
    
    return render_template("explore.html", **explore)
