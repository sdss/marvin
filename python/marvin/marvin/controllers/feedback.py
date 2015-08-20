#!/usr/bin/python

import os

import flask, sqlalchemy, json
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import text
from flask import request, render_template, send_from_directory, current_app, jsonify, Response
from flask import session as current_session
from manga_utils import generalUtils as gu
from collections import defaultdict, OrderedDict

from ..model.database import db

try: from inspection.marvin import Inspection
except: from marvin.inspection import Inspection

try:
    from . import valueFromRequest, processRequest
except ValueError:
    pass 

feedback_page = flask.Blueprint("feedback_page", __name__)

@feedback_page.route('/marvin/feedback.html', methods=['GET','POST'])
@feedback_page.route('/feedback.html', methods=['GET','POST'])
def feedback():
    ''' User feedback page '''
    
    feedback = {}
    feedback['title'] = "Marvin | Feedback"
    
    # get inspection
    feedback['inspection'] = inspection = Inspection(current_session)
    feedback['ready'] = inspection.ready
    
    # build products
    #inspection.component = OrderedDict([('Marvin','marvin'),('DRP','mangadrp'),('DAP','mangadap'),('Mavis','mangacas')])
    inspection.set_component()
    feedback['products'] = inspection.component.keys()
    
    # build types
    #inspection.type = OrderedDict([('Feature Request','enhancement'), ('Bug','defect'), ('Use Case','task'), ('Other','task')])
    inspection.set_type()
    feedback['types'] = inspection.type.keys()
    
    # get form feedback and add to db
    if inspection.ready:
        addfeedback = valueFromRequest(key='feedback_form',request=request, default=None)
    
        # add feedback to db
        if addfeedback:
            form = processRequest(request=request)
            feedback['form'] = form
            inspection.submit_feedback(form=form)
    inspection.retrieve_feedbacks()
    result = inspection.result()

    print('feedback statuses in inspection',inspection.feedbackstatuses)
    print('feedback table in inspection', inspection.feedbacks)
    
    return render_template('feedback.html',**feedback)

@feedback_page.route('/marvin/feedback/tracticket/promote', methods=['GET','POST'])
@feedback_page.route('/feedback/tracticket/promote', methods=['GET','POST'])
def promotetracticket():
    ''' User feedback function to promote tracticket '''
    
    # get inspection
    inspection = Inspection(current_session)
    
    # get id from button and promote trac ticket
    if inspection.ready:
        id = valueFromRequest(key='id',request=request, default=None)
    
        # add feedback to db
        if id:
            inspection.set_feedback(id=id)
            inspection.promote_tracticket()

    result = inspection.result()
    
    if inspection.ready: current_app.logger.warning('Inspection> Trac Ticket {0}'.format(result))
    else: current_app.logger.warning('Inspection> FAILED PROMOTE Trac Ticket {0}'.format(result))

    return jsonify(result=result)

@feedback_page.route('/marvin/feedback/status/update', methods=['GET','POST'])
@feedback_page.route('/feedback/status/update', methods=['GET','POST'])
def updatefeedbackstatus():
    ''' User feedback function to update status '''
    
    # get inspection
    inspection = Inspection(current_session)
    
    # get id from button and update feedback with status
    if inspection.ready:
        id = valueFromRequest(key='id',request=request, default=None)
        status = valueFromRequest(key='status',request=request, default=None)
    
        print('inside feedback status', id, status)

        # add feedback to db
        if id:
            inspection.set_feedback(id=id)
            inspection.update_feedback(status=status)

    result = inspection.result()
    print('feedback status',result)

    if inspection.ready: current_app.logger.warning('Inspection> Feedback Status Update {0}'.format(result))
    else: current_app.logger.warning('Inspection> FAILED FEEDBACK STATUS UPDATE {0}'.format(result))

    return jsonify(result=result)

@feedback_page.route('/marvin/feedback/vote/update', methods=['GET','POST'])
@feedback_page.route('/feedback/vote/update', methods=['GET','POST'])
def updatefeedbackvote():
    ''' User feedback function to upvote/novote/downvote '''
    
    # get inspection
    inspection = Inspection(current_session)

    # get id from button and vote in [-1,0,1]
    if inspection.ready:
        id = valueFromRequest(key='id',request=request, default=None)
        vote = valueFromRequest(key='vote',request=request, default=None)
    
        print('inside feedback vote', id, vote)

        # add feedback to db
        if id and vote:
            inspection.set_feedback(id=id)
            inspection.vote_feedback(vote=vote)

    result = inspection.result()
    print('feedback vote',result)
    
    if inspection.ready: current_app.logger.warning('Inspection> Feedback Vote {0}'.format(result))
    else: current_app.logger.warning('Inspection> FAILED FEEDBACK VOTE {0}'.format(result))

    return jsonify(result=result)

