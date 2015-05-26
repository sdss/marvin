#!/usr/bin/python

import flask, sqlalchemy, json, os, glob
from flask import request, render_template, send_from_directory, current_app, session as current_session,jsonify
from ..model.database import db
from ..utilities import processTableData, getImages, getDAPImages,buildDAPFormDict

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try:
    from . import valueFromRequest,processRequest
except ValueError:
    pass

try: from inspection.marvin import Inspection
except: from marvin.inspection import Inspection
from hashlib import md5


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
    tags = valueFromRequest(key='tags',request=request, default=None)
    tags = json.loads(tags) if tags else []

    form={}
    form['plateid'] = plateid
    form['version'] = version
    form['ifuname'] = ifuname
    form['cubepk'] = cubepk
    form['comments'] = comments
    form['issueids'] = issueids
    form['tags'] = tags

    return form    


@comment_page.route('/marvin/addcomment', methods=['GET','POST'])
@comment_page.route('/addcomment', methods=['GET','POST'])
def addComment():
    '''  add comments into the inspection database '''
    
    current_app.logger.warning("REQUEST.FORM ==> %r" % request.form if request.method == 'POST' else "REQUEST.ARGS ==> %r" % request.args)
    session = db.Session()
    
    # grab form data
    form = getCommentForm()

    # add new comment to database
    inspection = Inspection(current_session)
    if inspection.ready:
        inspection.set_version(version=form['version'])
        inspection.set_ifudesign(plateid=form['plateid'],ifuname=form['ifuname'])
        inspection.set_cube(cubepk=form['cubepk'])
        inspection.submit_comments(comments=form['comments'],issueids=form['issueids'],tags=form['tags'])
    result = inspection.result()
    
    if inspection.ready: current_app.logger.warning('Inspection> ADD comments={0} with issueids={1}: {2}'.format(form['comments'],form['issueids'],result))
    else: current_app.logger.warning('Inspection> NOT READY TO ADD comments={0} with issueids={1}: {2}'.format(form['comments'],form['issueids'],result))

    if inspection.ready: current_app.logger.warning('Inspection> ADD result: {0}'.format(result))
    else: current_app.logger.warning('Inspection> NOT READY TO ADD result: {0}'.format(result))

    return jsonify(result=result)

@comment_page.route('/marvin/getcomment', methods=['GET','POST'])
@comment_page.route('/getcomment', methods=['GET','POST'])    
def getComment(all=None, cube=None):
    ''' get comments for a specific cube and user '''
    
    current_app.logger.warning("REQUEST.FORM ==> %r" % request.form if request.method == 'POST' else "REQUEST.ARGS ==> %r" % request.args)
    session = db.Session()
    
    # grab form info
    form = getCommentForm()
    
    # grab info from cube
    if cube:
        form['cubepk'] = cube.pk
        form['plateid'] = cube.plate
        form['ifuname'] = cube.ifu.name
        form['version'] = cube.pipelineInfo.version.version
    
    # grab comments for user and cube , if none returns empty list  
    # if all=True set, then grab all comments from all users for this cube  
    
    inspection = Inspection(current_session)
    if inspection.ready:
        inspection.set_version(version=form['version'])
        inspection.set_ifudesign(plateid=form['plateid'],ifuname=form['ifuname'])
        inspection.set_cube(cubepk=form['cubepk'])
        inspection.retrieve_comments()
        inspection.retrieve_alltags()
        inspection.retrieve_tags()
    result = inspection.result()
    #if '12701' in form['ifuname']: result['tags']=['hello','world']

    if inspection.ready: current_app.logger.warning('Inspection> GET comments/issueids: {0}'.format(result))
    else: current_app.logger.warning('Inspection> NOT READY TO GET comments/issueids: {0}'.format(result))
    
    return jsonify(result=result)
    
@comment_page.route('/marvin/login', methods=['GET','POST'])
@comment_page.route('/login', methods=['GET','POST'])      
def login():
    ''' login for trac user '''

    username = valueFromRequest(key='username',request=request, default=None)
    password = valueFromRequest(key='password',request=request, default=None)
    auth = md5("%s:AS3Trac:%s" % (username.strip(),password.strip())).hexdigest() if username and password else None
    inspection = Inspection(current_session, username=username, auth=auth)
    if not inspection.ready: current_app.logger.warning('Inspection> NOT READY: %r %r' % (username,auth))
    result = inspection.result()
    
    if inspection.ready: current_app.logger.warning('Inspection> SUCCESSFUL LOGIN {0}'.format(result))
    else: current_app.logger.warning('Inspection> FAILED LOGIN {0}'.format(result))

    return jsonify(result=result)

# DAP Plot Panel Retrieval    
@comment_page.route('/marvin/getdappanel', methods=['GET','POST'])
@comment_page.route('/getdappanel', methods=['GET','POST'])
def getdappanel():
    ''' Retrieve a DAP QA panel of plots based on form data'''
    
    current_app.logger.warning("REQUEST.FORM ==> %r" % request.form if request.method == 'POST' else "REQUEST.ARGS ==> %r" % request.args)
    dapform = processRequest(request=request)

    # store form in session, using old mapid, qatype
    setSessionDAPComments(dapform) # old mapid 
    
    # Get real plots
    mode,bintype = dapform['qatype'].split('-')
    imglist,msg = getDAPImages(dapform['plateid'], dapform['ifu'], dapform['drpver'], 
        dapform['dapver'], dapform['key'], mode, bintype, dapform['mapid'])
        
    # Build title
    qatype = '-'.join([x.upper() for x in dapform['qatype'].split('-')])
    defaulttitle = {'maps':'Maps','radgrad':'Radial Gradients','spectra':'Spectra'}
    if dapform['key'] != 'spectra':
        maptype = {'kin':'Kinematic','snr':'SNR','binnum':'Bin_Num','emflux':'EMflux','emfluxew':'EMflux_EW','emfluxfb':'EMflux_FB'}
        newtitle = '{1}: {2}-{0}'.format(maptype[dapform['mapid']],defaulttitle[dapform['key']],qatype) if dapform['mapid'] in maptype else defaulttitle[dapform['key']]
    else:
        name = 'spec-{0:04d}'.format(int(dapform['mapid'].split('c')[1]))
        newtitle = '{1}: {2}-{0}'.format(name,defaulttitle[dapform['key']],qatype)
    
    # load new form from session, using current mapid, qatype
    getSessionDAPComments(dapform) # current mapid
         
    result={}
    result['title'] = newtitle
    result['images'] = imglist if imglist else None
    result['status'] = 0 if not imglist else 1
    result['msg'] = msg
    
    return jsonify(result=result)

@comment_page.route('/marvin/getdapspeclist', methods=['POST'])   
@comment_page.route('/getdapspeclist', methods=['POST'])   
def getdapspeclist():
    ''' Retrieve a list of DAP spectra for a given set of inputs '''
    
    current_app.logger.warning("REQUEST.FORM ==> %r" % request.form if request.method == 'POST' else "REQUEST.ARGS ==> %r" % request.args)
    dapform = processRequest(request=request)

    # get real plots
    mode,bintype = dapform['qatype'].split('-')
    imglist,msg = getDAPImages(dapform['plateid'], dapform['ifu'], dapform['drpver'], 
        dapform['dapver'], dapform['key'], mode, bintype, dapform['mapid'],filter=False)
    
    # extract spectra names
    if imglist:
        speclist = [i.rsplit('_',1)[1].split('.')[0] for i in imglist]
    else: speclist = None

    result={}
    result['speclist'] = speclist 
    result['status'] = 0 if not imglist else 1
    result['msg'] = msg
    
    return jsonify(result=result)


def setSessionDAPComments(form):
    ''' store session dap comments based on form input, uses oldmapid 
    
    	form info
    		comment syntax: dapqa_comment{categoryid}_{mapnumber} - from all cat. divs
    		issue syntax: "issue{issueid}_{mapnumber} - only from given cat. div at a time
    ''' 

    if not form['oldkey']: form['oldkey'] = form['key']
    if not form['oldmapid']: form['oldmapid'] = form['mapid']
    if not form['oldqatype']: form['oldqatype'] = form['qatype']
    
    # build a blank new form dict
    formdict = buildDAPFormDict(form)
    
    # populate appropriate point with comments/issues
    catkey={'maps':'1','radgrad':'2','spectra':'3'}
    cattype,bin = form['oldqatype'].split('-')
    comments = [val for key,val in form.iteritems() if 'dapqa_comment'+catkey[form['oldkey']] in key]
    # make issues array of int tuples (issueid,mapnumber)
    issues = json.loads(form['issues'])
    issues = [(int(i.rsplit('_')[1]),int(i.rsplit('_')[2])) for i in issues] if 'any' not in issues else []
    # add to formdict
    formdict[form['oldkey']][cattype][bin][form['oldmapid']]['issues'] = issues
    formdict[form['oldkey']][cattype][bin][form['oldmapid']]['comments'] = comments
    
    print('orig form', form)
    print(' ')
    print('formdict', formdict)
    
def getSessionDAPComments(form):
    ''' retrieve session dap comments based on form input, uses newmapid '''
    
    # build a blank new form dict
    formdict = buildDAPFormDict(form)
    
    
    
    
        
    
        