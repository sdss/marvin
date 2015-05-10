#!/usr/bin/python

import flask, sqlalchemy, json
from flask import request, render_template, send_from_directory, current_app, session as current_session,jsonify
from ..model.database import db
from ..utilities import processTableData

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try:
    from . import valueFromRequest
except ValueError:
    pass

try: from inspection.manga import Feedback
except: from mangasas.inspection import Feedback

comment_page = flask.Blueprint("comment_page", __name__)

def getCommentForm():
    ''' get the form data form the comment box'''

    plateid = valueFromRequest(key='plateid',request=request, default=None)
    version = valueFromRequest(key='version',request=request, default=None)
    ifuname = valueFromRequest(key='ifuname',request=request, default=None)        
    cubepk = valueFromRequest(key='cubepk',request=request, default=None)        
    comments = valueFromRequest(key='comments',request=request, default=None)
    comments = json.loads(comments) if comments else []
    issueids = valueFromRequest(key='issueids',request=request, default=None)
    issueids = json.loads(issueids) if issueids else []
    
    form={}
    form['plateid'] = plateid
    form['version'] = version
    form['ifuname'] = ifuname
    form['cubepk'] = cubepk
    form['comments'] = comments
    form['issueids'] = issueids

    return form    


@comment_page.route('/manga/addcomment', methods=['GET','POST'])
@comment_page.route('/addcomment', methods=['GET','POST'])
def addComment():
    '''  add comments into the inspection database '''
    
    current_app.logger.debug("REQUEST.FORM ==> %r" % request.form if request.method == 'POST' else "REQUEST.ARGS ==> %r" % request.args)
    session = db.Session()
    
    # grab form data
    form = getCommentForm()
    print('version',form['version'])
    print('cube', form['cubepk'])    
        
    # add new comment to database
    feedback = Feedback(current_session, username='sdss', auth='sdss')
    if feedback.ready:
        feedback.set_version(version=form['version'])
        feedback.set_cube(plateid=form['plateid'],ifuname=form['ifuname'],cube_pk=form['cubepk'])
        feedback.submit_comments(comments=form['comments'],issueids=form['issueids'])
    result = feedback.result()
    if feedback.ready: current_app.logger.info('Feedback> ADD comments={0} with issueids={1}: {2}'.format(form['comments'],form['issueids'],result))
    else: current_app.logger.warning('Feedback> NOT READY TO ADD comments={0} with issueids={1}'.format(form['comments'],form['issueids']))
    
    print('result')
    
    return jsonify(result=result)

@comment_page.route('/manga/getcomment', methods=['GET','POST'])
@comment_page.route('/getcomment', methods=['GET','POST'])    
def getComment(all=None, cube=None):
    ''' get comments for a specific cube and user '''
    
    current_app.logger.debug("REQUEST.FORM ==> %r" % request.form if request.method == 'POST' else "REQUEST.ARGS ==> %r" % request.args)
    session = db.Session()
    
    # grab form info
    form = getCommentForm()
    print('version',form['version'])
    print('cube', form['cubepk'])
    
    # grab info from cube
    if cube:
        form['cubepk'] = cube.pk
        form['plateid'] = cube.plate
        form['ifuname'] = cube.ifu.name
        form['version'] = cube.pipelineInfo.version.version
    
    # grab comments for user and cube , if none returns empty list  
    # if all=True set, then grab all comments from all users for this cube  
    
    # User Info
    userid = 'sdss'
    auth='sdss'

    feedback = Feedback(current_session, username=userid, auth=auth)
    if feedback.ready:
        feedback.set_version(version=form['version'])
        feedback.set_cube(plateid=form['plateid'],ifuname=form['ifuname'],cube_pk=form['cubepk'])
        feedback.set_comments()
    result = feedback.result()
    if feedback.ready: current_app.logger.info('Feedback> GET comments/issueids: {0}'.format(result))
    else: current_app.logger.info('Feedback> NOT READY TO GET comments/issueids')
    
    # add some temp testing info
    result['userid'] = 'bac29'
    result['cubepk'] = form['cubepk']
    result['membername'] = 'Brian Cherinka'
    result['comments'] = {'1':[],'2':['comment1','comment2','comment3'],'3':['commentA','commentB','commentC'],'4':['comment7','comment8','comment9']}
    print('result',result)
    
    return jsonify(result=result)

@comment_page.route('/manga/login', methods=['GET','POST'])
@comment_page.route('/login', methods=['GET','POST'])      
def login():
    ''' login for trac user '''

    username = valueFromRequest(key='username',request=request, default=None)
    password = valueFromRequest(key='password',request=request, default=None)
     
    print(username,password) 
     
    return jsonify(result='logging in')    
    
    