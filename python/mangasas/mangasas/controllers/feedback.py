#!/usr/bin/python

import os

import flask, sqlalchemy, json
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import text
from flask import request, render_template, send_from_directory, current_app, jsonify, Response
from manga_utils import generalUtils as gu
from collections import defaultdict
import numpy as np

from ..model.database import db
from ..utilities import makeQualNames, processTableData

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try:
    from . import valueFromRequest
except ValueError:
    pass 

feedback_page = flask.Blueprint("feedback_page", __name__)

@feedback_page.route('/feedback.html', methods=['GET','POST'])
def feedback():
    ''' User feedback page '''
    
    feedback = {}
    
    # build types
    types = ['Feature Request', 'Bug', 'Use Case', 'Other']
    feedback['types'] = types
    
    # get form feedback
    addfeedback = valueFromRequest(key='feedback_form',request=request, default=None)
    type = valueFromRequest(key='feedbacktype',request=request, default=None)
    text = valueFromRequest(key='feedbackfield',request=request, default=None)
    
    # add feedback to db
    if addfeedback:
        pass
    
	# get feedback from db
	results = None
	feedback['table'] = results
    
    return render_template('feedback.html',**feedback)
