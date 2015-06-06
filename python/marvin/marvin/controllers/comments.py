#!/usr/bin/python

import flask, sqlalchemy, json, os, glob
from flask import request, render_template, send_from_directory, current_app, session as current_session,jsonify
from ..model.database import db
from ..utilities import getImages, getDAPImages

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
        inspection.set_drpver(drpver=form['version'])
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
        inspection.set_drpver(drpver=form['version'])
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
    dapform['tags'] = json.loads(dapform['tags']) if 'tags' in dapform else []
    
    print('first dapform',dapform)

    # store form in session, using old mapid, qatype, and key
    setresults = setSessionDAPComments(dapform) if any([dapform['oldmapid'],dapform['oldkey'],dapform['oldqatype']]) else None
    
    # Get real plots
    mode,bintype = dapform['qatype'].split('-')
    imglist,msg = getDAPImages(dapform['plateid'], dapform['ifu'], dapform['drpver'], 
        dapform['dapver'], dapform['key'], mode, bintype, dapform['mapid'])
    print('final images',imglist)    
    # Build title
    qatype = '-'.join([x.upper() for x in dapform['qatype'].split('-')])
    defaulttitle = {'maps':'Maps','radgrad':'Radial Gradients','spectra':'Spectra'}
    if dapform['key'] != 'spectra':
        maptype = {'kin':'Kinematic','snr':'SNR','binnum':'Bin_Num','emflux':'EMflux','emfluxew':'EMflux_EW','emfluxfb':'EMflux_FB'}
        newtitle = '{1}: {2}-{0}'.format(maptype[dapform['mapid']],defaulttitle[dapform['key']],qatype) if dapform['mapid'] in maptype else defaulttitle[dapform['key']]
    else:
        name = 'spec-{0:04d}'.format(int(dapform['mapid'].split('c')[1]))
        newtitle = '{1}: {2}-{0}'.format(name,defaulttitle[dapform['key']],qatype)
    
    # load new form from session, using current mapid, qatype, and key
    getresults = getSessionDAPComments(dapform) 
    result={}
    result['title'] = newtitle
    result['images'] = imglist if imglist else None
    result['status'] = 0 if not imglist else 1
    result['panelmsg'] = msg
    result['setsession'] = setresults
    result['getsession'] = getresults
    
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
    		issue syntax: "issue_{issueid}_{mapnumber} - only from given cat. div at a time
    ''' 
    
    # set default old values if they are empty 
    if not form['oldkey']: form['oldkey'] = form['key']
    if not form['oldmapid']: form['oldmapid'] = form['mapid']
    if not form['oldqatype']: form['oldqatype'] = form['qatype']

    print('inside setsession: form', form)
    
    # populate appropriate point with comments/issues
    inspection = Inspection(current_session)
    catkey = {val['key']:key for key,val in inspection.dapqacategory.iteritems()}
    mode,bin = form['oldqatype'].split('-')
    sortedcomments = sorted([(key,val) for key,val in form.iteritems() if 'dapqa_comment'+catkey[form['oldkey']] in key])
    comments = [comment[1] for comment in sortedcomments]
    # get issues, separate into ints by panel below
    issues = json.loads(form['issues'])
    # make panel names
    if 'emflux' in form['oldmapid']: 
        panelname = ['oii','hbeta','oiii','halpha','nii','sii']
    elif 'snr' in form['oldmapid']:
        panelname = ['signal','noise','snr','halpha_ew','resid','chisq']
    elif 'kin' in form['oldmapid']:
        if 'ston' in bin: panelname = ['emvel','emvdisp','sth3','stvel','stvdisp','sth4']
        elif 'none' in bin: panelname = ['emvel','emvdisp','chisq','stvel','stvdisp','resid']
    else: panelname = ['spectrum']
    
    # build panel info list
    catid = catkey[form['oldkey']]
    panelcomments = [{'panel':name,'position':i+1,'catid':catid,'comment':comments[i],
    'issues':[int(iss.split('_')[1]) for iss in issues if iss.rsplit('_')[-1] == str(i+1)]} for i,name in enumerate(panelname)]
    
    print('panelcomment', panelcomments)
    
    # add new comment to database
    if inspection.ready:
        inspection.set_version(drpver=form['drpver'],dapver=form['dapver'])
        inspection.set_ifudesign(plateid=form['plateid'],ifuname=form['ifu'])
        inspection.set_cube(cubepk=form['cubepk'])
        inspection.set_option(mode=mode,bintype=bin,maptype=form['oldmapid'])
        inspection.set_session_dapqacomments(catid=catid,comments=panelcomments,touched=True)
        inspection.set_session_tags(tags=form['tags'])
        if 'dapqacomments' in current_session: print("setSessionDAPComments -> current_session['dapqacomments']=%r" % current_session['dapqacomments'])
        if 'submit' in form and form['submit']: inspection.submit_dapqacomments()
        if 'reset' in form and form['reset']:
            inspection.drop_dapqacomments_from_session()
            inspection.drop_dapqatags_from_session()
    result = inspection.result()

    if inspection.ready: current_app.logger.warning('Inspection> set DAPQA Comments {0}'.format(result))
    else: current_app.logger.warning('Inspection> FAILED to set DAPQA Comments {0}'.format(result))

    return result

    
def getSessionDAPComments(form):
    ''' retrieve session dap comments based on form input, uses newmapid '''
    
    # new key,map information
    inspection = Inspection(current_session)
    catkey = {val['key']:key for key,val in inspection.dapqacategory.iteritems()}
    mode,bin = form['qatype'].split('-')
    maptype = form['mapid']
    catid = catkey[form['key']]
    
    # get comments from database
    if inspection.ready:
        inspection.set_version(drpver=form['drpver'],dapver=form['dapver'])
        inspection.set_ifudesign(plateid=form['plateid'],ifuname=form['ifu'])
        inspection.set_cube(cubepk=form['cubepk'])
        inspection.set_option(mode=mode,bintype=bin,maptype=maptype)
        inspection.retrieve_dapqacomments(catid=catid)
    result = inspection.result()
    
    if inspection.ready: current_app.logger.warning('Inspection> get DAPQA Comments {0}'.format(result))
    else: current_app.logger.warning('Inspection> FAILED to get DAPQA Comments {0}'.format(result))

    return result    
    
    
    
    
    
        
    
        