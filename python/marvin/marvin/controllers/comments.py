#!/usr/bin/python

import flask, sqlalchemy, json, os, glob, warnings, sys
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
from itertools import groupby


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

# split the DAP qatype
def splitQAType(qatype):
    ''' split the DAP qatype variable into mode (cube/rss) and bintype'''
    try:
        mode,bintype = qatype.split('-')
    except ValueError as e:
        raise ValueError('ValueError: {0}'.format(e))
    
    return mode, bintype


# DAP Plot Panel Retrieval    
@comment_page.route('/marvin/getdappanel', methods=['POST'])
@comment_page.route('/getdappanel', methods=['POST'])
def getdappanel():
    ''' Retrieve a DAP QA panel of plots based on form data'''

    result={}    
    current_app.logger.warning("REQUEST.FORM ==> %r" % request.form if request.method == 'POST' else "REQUEST.ARGS ==> %r" % request.args)
    dapform = processRequest(request=request)
    dapform['tags'] = json.loads(dapform['tags']) if 'tags' in dapform else []
    inspection = Inspection(current_session)
    
    print('first dapform',dapform)

    # store form in session, using old mapid, qatype, and key
    try: 
        setresults = setSessionDAPComments(dapform) if any([dapform['oldmapid'],dapform['oldkey'],dapform['oldqatype']]) else None
    except RuntimeError as error:
        result['status'] = -1
        result['setsession']={'status':-1, 'message':'Error in setSessionDAPComments: {0}'.format(error)}
        result['panelmsg'] = result['setsession']['message'];
        return jsonify(result=result)

    # split the QA type into cube/rss mode and bintype
    try:
        mode,bintype = splitQAType(dapform['qatype'])
    except ValueError as e:
        result['status'] = -1
        msg = 'Error splitting qatype {0}: {1}'.format(dapform['qatype'],e)
        result['panelmsg'] = msg
        warnings.warn(msg,RuntimeWarning)
        return jsonify(result=result)

    # Get real plots
    imglist,msg = getDAPImages(dapform['plateid'], dapform['ifu'], dapform['drpver'], 
        dapform['dapver'], dapform['key'], mode, bintype, dapform['mapid'], dapform['specpanel'],inspection)

    # Get panel names
    if bintype:
        names = inspection.get_panelnames(dapform['mapid'],bin=bintype) if dapform['key'] != 'spectra' else inspection.get_panelnames('specmap',bin=bintype)
        panelnames = [name[1] for name in names]

    # Build title
    qatype = dapform['qatype'].upper()
    defaulttitle = inspection.dapqaoptions['defaulttitle']
    if dapform['key'] != 'spectra':
        maptype = inspection.dapqaoptions['maptype']
        newtitle = '{1}: {2}-{0}'.format(maptype[dapform['mapid']],defaulttitle[dapform['key']],qatype) if dapform['mapid'] in maptype else defaulttitle[dapform['key']]
    else:
        name = 'spec-{0:04d}'.format(int(dapform['mapid'].split('c')[1]))
        newtitle = '{1}: {2}-{0}'.format(name,defaulttitle[dapform['key']],qatype)
    
    # load new form from session, using current mapid, qatype, and key
    try:
        getresults = getSessionDAPComments(dapform)
    except RuntimeError as error: 
        result['status'] = -1
        result['getsession']={'status':-1, 'message':'Error in getSessionDAPComments: {0}'.format(error)}
        result['panelmsg'] = result['getsession']['message'];
        return jsonify(result=result)

    result['title'] = newtitle
    result['images'] = imglist if imglist else None
    result['panels'] = panelnames
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
    inspection = Inspection(current_session)

    # split the QA type into cube/rss mode and bintype
    try:
        mode,bintype = splitQAType(dapform['qatype'])
    except ValueError as e:
        result['status'] = -1
        msg = 'Error splitting qatype {0}: {1}'.format(dapform['qatype'],e)
        result['msg'] = msg
        warnings.warn(msg,RuntimeWarning)
        return jsonify(result=result)

    # get real plots
    imglist,msg = getDAPImages(dapform['plateid'], dapform['ifu'], dapform['drpver'], 
        dapform['dapver'], dapform['key'], mode, bintype, dapform['mapid'], dapform['specpanel'],inspection, filter=False)

    # extract spectra names
    if imglist:
        speclist = [i.rsplit('_',1)[1].split('.')[0] for i in imglist if 'spec-' in i.rsplit('_',1)[1]]
        speclist.sort()
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
    
    result={}

    # set default old values if they are empty 
    if not form['oldkey'].strip(): form['oldkey'] = form['key']
    if not form['oldmapid'].strip(): form['oldmapid'] = form['mapid']
    if not form['oldqatype'].strip(): form['oldqatype'] = form['qatype']

    # populate appropriate point with comments/issues
    inspection = Inspection(current_session)
    catkey = {val['key']:key for key,val in inspection.dapqacategory.iteritems()}

    # split the QA type into cube/rss mode and bintype
    try:
        mode,bin = splitQAType(form['oldqatype'])
    except ValueError as e:
        result['status'] = -1
        result['message'] = 'Error splitting old qatype {0}: {1}'.format(form['oldqatype'],e)
        raise RuntimeError(result['message'])

    # grab the right panel comments and generate single array of panel comments
    try:
        if 'binnum' in form['oldmapid']:
            sortedcomments = sorted([(key,val) for key,val in form.iteritems() if 'dapqa_comment{0}_binnum'.format(catkey[form['oldkey']]) in key])
        elif 'spectra' in form['oldkey'] and 'single' in form['oldspecpanel']:
            sortedcomments = sorted([(key,val) for key,val in form.iteritems() if 'dapqa_comment{0}_single'.format(catkey[form['oldkey']]) in key])
        else:
            sortedcomments = sorted([(key,val) for key,val in form.iteritems() if 'dapqa_comment{0}'.format(catkey[form['oldkey']]) in key and key.split('_')[2].isdigit()])
        comments = [comment[1] for comment in sortedcomments]
    except:
        result['status'] = -1
        result['message'] = 'Error sorting comments with old key {0},mapid {1},specpanel {2}: {3}'.format(form['oldkey'],form['oldmapid'],form['oldspecpanel'],sys.exc_info()[1])
        raise RuntimeError(result['message'])        

    # get issues, separate into ints by panel below
    issues = json.loads(form['issues'])
    try: 
        issues = {key: list(int(iss.split('_')[1]) for iss in val) for key, val in groupby(issues, key=lambda x: x.split('_')[-1])} if type(issues)==list else issues
    except:
        result['status'] = -1
        result['message'] = 'Error separating issues {0}: {1}'.format(issues,sys.exc_info()[1])
        raise RuntimeError(result['message'])

    # make panel names
    panel_input = 'specmap' if form['oldspecpanel'] != 'single' and form['oldkey'] == 'spectra' else form['oldmapid']
    try:
        names =  inspection.get_panelnames(panel_input,bin=bin)
        panelname = [name[1] for name in names]
    except:
        result['status'] = -1
        result['message'] = 'Error getting panel names from inspection with panel input {0}, bin {1}: {2} {3}'.format(panel_input,bin,sys.exc_info()[0],sys.exc_info()[1])
        raise RuntimeError(result['message'])   
    
    # build panel info list
    catid = catkey[form['oldkey']]
    panelcomments = []
    try:
        for i,name in enumerate(panelname):
            if issues == 'any': panelissues = []
            elif 'binnum' in issues: panelissues = issues['binnum']
            elif 'single' in issues: panelissues = issues['single']
            else: panelissues = issues[str(i+1)] if str(i+1) in issues else []

            tmp = {'panel':name, 'position':i+1,'catid':catid,'comment':comments[i],'issues':panelissues}
            panelcomments.append(tmp)
    except:
        result['status'] = -1
        result['message'] = 'Error building panel comments for inspection: {0}, {1}'.format(sys.exc_info()[0],sys.exc_info()[1])
        raise RuntimeError(result['message'])   
    
    print('inside set session panelcomment', panelcomments)

    # add new comment to database
    if inspection.ready:
        inspection.set_version(drpver=form['drpver'],dapver=form['dapver'])
        inspection.set_ifudesign(plateid=form['plateid'],ifuname=form['ifu'])
        inspection.set_cube(cubepk=form['cubepk'])
        inspection.set_option(mode=mode,bintype=bin,maptype=form['oldmapid'])
        try:  
            inspection.set_session_dapqacomments(catid=catid,comments=panelcomments,touched=True)
            inspection.set_session_tags(tags=form['tags'])
        except:
            result['status'] = -1
            result['message'] = 'Error setting session comments for inspection: {0}, {1}'.format(sys.exc_info()[0],sys.exc_info()[1])
            raise RuntimeError(result['message'])   

        #if 'dapqacomments' in current_session: print("setSessionDAPComments -> current_session['dapqacomments']=%r" % current_session['dapqacomments'])
        try:
            if 'submit' in form and form['submit']: inspection.submit_dapqacomments()
        except:
            result['status'] = -1
            result['message'] = 'Error submitting comments to inspection: {0}, {1}'.format(sys.exc_info()[0],sys.exc_info()[1])
            raise RuntimeError(result['message'])   

        try:
            if 'reset' in form and form['reset']:
                inspection.drop_dapqacomments_from_session()
                inspection.drop_dapqatags_from_session()
        except:
            result['status'] = -1
            result['message'] = 'Error resetting session comments for inspection: {0}, {1}'.format(sys.exc_info()[0],sys.exc_info()[1])
            raise RuntimeError(result['message'])   

    result = inspection.result()

    if inspection.ready: current_app.logger.warning('Inspection> set DAPQA Comments {0}'.format(result))
    else: current_app.logger.warning('Inspection> FAILED to set DAPQA Comments {0}'.format(result))

    return result

    
def getSessionDAPComments(form):
    ''' retrieve session dap comments based on form input, uses newmapid '''
    
    result={}
    # new key,map information
    inspection = Inspection(current_session)
    maptype = form['mapid']
    try: 
        catkey = {val['key']:key for key,val in inspection.dapqacategory.iteritems()}
        catid = catkey[form['key']]
    except:
        result['status'] = -1
        result['message'] = 'Error parsing inspection categories: {0}'.format(sys.exc_info()[1])
        raise RuntimeError(result['message'])          

    # split the QA type into cube/rss mode and bintype
    try:
        mode,bin = splitQAType(form['qatype'])
    except ValueError as e:
        result['status'] = -1
        result['message'] = 'Error splitting qatype {0}: {1}'.format(form['qatype'],e)
        raise RuntimeError(result['message'])

    # get comments from database
    if inspection.ready:
        inspection.set_version(drpver=form['drpver'],dapver=form['dapver'])
        inspection.set_ifudesign(plateid=form['plateid'],ifuname=form['ifu'])
        inspection.set_cube(cubepk=form['cubepk'])
        inspection.set_option(mode=mode,bintype=bin,maptype=maptype)
        if form['specpanel'] == 'single': inspection.single=True
        inspection.retrieve_dapqacomments(catid=catid)
        inspection.retrieve_alltags()
    result = inspection.result()

    # set up binnum and single spectra positions
    if 'dapqacomments' in result:
        if form['mapid'] == 'binnum': result['dapqacomments'][0]['position'] = 'binnum'
        if form['key'] == 'spectra' and form['specpanel'] == 'single': result['dapqacomments'][0]['position'] = 'single'
        
    if inspection.ready: current_app.logger.warning('Inspection> get DAPQA Comments {0}'.format(result))
    else: current_app.logger.warning('Inspection> FAILED to get DAPQA Comments {0}'.format(result))

    return result    
    
    
    
    
    
        
    
        