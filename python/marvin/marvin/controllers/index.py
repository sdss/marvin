#!/usr/bin/python

import os, glob, random

import flask
from flask import request, render_template, send_from_directory, current_app, session as current_session, jsonify
from manga_utils import generalUtils as gu
from collections import OrderedDict
from sqlalchemy import func

from ..model.database import db
from ..utilities import getImages, testDBConnection, configFeatures, setGlobalSession, updateGlobalSession
from ..jinja_filters import getMPL

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try:
    from . import valueFromRequest,processRequest
except ValueError:
    pass

def getDblist(param):
    ''' Get a list of specified parameters from db '''

    session = db.Session()
    try:
        if param == 'plateid': items = session.query(datadb.Cube.plate).all()
        if param == 'mangaid': items = session.query(datadb.Cube.mangaid).all()
    except:
        raise RuntimeError('Error querying database with param {0}'.format(param))

    options = [str(item[0]) for item in set(items)]
    options.sort()

    return options

def checkForm(form):
    '''
      checks the form against the specified keys, check if key exists in form, and if form[key] has a value 
    '''
    keys = ['vermode', 'version', 'dapversion', 'mplver']
    inform = [key in form and form[key] for key in keys]
    return all(inform)
    
index_page = flask.Blueprint("index_page", __name__)

@index_page.route('/setsearch/', methods=['GET','POST'])
@index_page.route('/marvin/setsearch/', methods=['GET','POST'])
def setsearch():
    ''' Set search mode and return typeahead options '''

    searchid = valueFromRequest(key='searchid',request=request, default=None)
    result={}
    if searchid:
        current_session['searchmode'] = searchid
        result['status'] = 1
        result['msg'] = 'Success'
        try:
            result['options'] = getDblist(searchid)
            #current_session['searchoptions'] = result['options']
            #print('new current session options',current_session['searchoptions'])
        except RuntimeError as e:
            result['options'] = None
            result['status'] = -1
            result['msg'] = 'Error getting options list: {0}'.format(e)
    else:
        result['status'] = -1
        result['msg'] = 'Error: search id is None.  Check javascript.'

    return jsonify(result=result)

@index_page.route('/setmode/',methods=['GET','POST'])
@index_page.route('/marvin/setmode/',methods=['GET','POST'])
def setmode():
    ''' Set mode of Marvin to use '''

    marvinmode = valueFromRequest(key='marvinmode',request=request, default=None)
    result={}
    result['status'] = 1
    result['msg'] = 'Success'

    if marvinmode == 'dr13':
        current_session['marvinmode']='dr13'
    elif marvinmode == 'mangawork':
        current_session['marvinmode']='mangawork'
    else:
        result['status'] = -1
        result['msg'] = 'Error setting mode: marvinmode is not expected value'

    # update global session
    updateGlobalSession(mode=marvinmode)

    # configure the feature tags
    configFeatures(current_app, marvinmode)

    return jsonify(result=result)

@index_page.route('/setversion/', methods=['GET','POST'])
@index_page.route('/marvin/setversion/', methods=['GET','POST'])
def setversion():
    ''' Set version to use during MaNGA SAS '''
    
    # get form    
    verform = processRequest(request=request)
    # check the form
    goodform = checkForm(verform)

    if goodform:
        # set version mode
        current_session['vermode'] = 'MPL' if verform['vermode']=='mpl' else 'DRP/DAP'
        
        # set version
        if verform['vermode']=='mpl':
            mpl = getMPL(verform['mplver'])
            drpver,dapver = mpl.split(':')[1].strip().split(', ')
            current_session['currentver'] = drpver
            current_session['currentdapver'] = dapver if dapver != 'NA' else None
            current_session['currentmpl'] = verform['mplver']
            msg = 'Success'
            status = 1
        elif verform['vermode']=='drpdap':
            current_session['currentver'] = verform['version']
            current_session['currentdapver'] = verform['dapversion']
            current_session['currentmpl'] = verform['mplver']
            msg = 'Success'
            status = 1
        else:
            current_app.logger.error('Error in setversion: vermode not set properly.  Please check the javascript.')
            msg = 'Error in setversion: vermode not set properly.  Please check the javascript.'
            status = -1
            
    else:
        current_app.logger.error('Error in setversion: verform is not a proper form.  Either missing key or empty string. Please check the javascript.')
        msg = 'Error in setversion: verform is not a proper form.  Either missing key or empty string. Please check the javascript.'
        status = -1

    result = {'name' : 'setversion', 'desc': 'sets the global version of Marvin'}
    result['status'] = status
    result['msg'] = msg
    result['data'] = verform

    return jsonify(result=result)

@index_page.route('/', methods=['GET'])
@index_page.route('/index/', methods=['GET'])
def index():
    ''' Documentation here. '''
    
    session = db.Session()    
    index = {}
    index['title'] = "Marvin"
    
    # test DB connection
    error = testDBConnection(session)
    if error:
        current_session['vermode']='MPL'
        return render_template('errors/bad_db_access.html', **{'message':error})

    # set global session variables
    setGlobalSession()
    version = current_session['currentver']

    current_app.logger.info('Loading index page...')
    
    # find and grab all images ; convert paths to sas paths
    redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version)
    good = os.path.isdir(redux)
    index['good'] = good
    images = getImages(version=version) 
    
    if any(images):        
        # randomize the images
        imgcount = 4
        if len(images) < imgcount: biglist = [random.choice(images) for i in xrange(imgcount)]
        else: biglist = random.sample(images,imgcount)
        images = biglist

        # make unique image dictionary
        imdict = OrderedDict()
        for image in biglist:
            plate = image.rsplit('/',4)[1]
            name = image.rsplit('/',4)[-1].split('.')[0]
            plateifu = '{0}-{1}'.format(plate,name)
            imdict[plateifu] = image
        index['imdict'] = imdict
    else: index['imdict'] = None
    
    # get all galaxy cubes from latest tag; get cube counts by plate
    cubequery = session.query(func.count(datadb.Cube.pk),datadb.Cube.plate).join(datadb.PipelineInfo,datadb.PipelineVersion,datadb.IFUDesign)\
        .filter(datadb.PipelineVersion.version==version,datadb.IFUDesign.nfiber != 7).group_by(datadb.Cube.plate)
    cubelist = cubequery.all()
    platecount = len(cubelist)
    cubecount = sum([cube[0] for cube in cubelist])
 
    # get all plate types (for right now..eventually build plate table and summary table)
    galcubequery = cubequery.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MNGTRG1',datadb.FitsHeaderValue.value != '0')
    starcubequery = cubequery.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MNGTRG2',datadb.FitsHeaderValue.value != '0')
    anccubequery = cubequery.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MNGTRG3',datadb.FitsHeaderValue.value != '0')   
    
    gallist = galcubequery.all()
    starlist = starcubequery.all()
    anclist = anccubequery.all()
    galplates = len(gallist)
    starplates = len(starlist)
    ancplates = len(anclist)
    galcubes = sum([cube[0] for cube in gallist])

    types = {'Stellar':starplates, 'Galaxy': galplates, 'Ancillary':ancplates}

    index['cubes'] = galcubes
    index['plates'] = platecount
    index['types'] = types
    index['labels'] = ['Galaxy', 'Stellar', 'Ancillary', None]
    index['modnum'] = imgcount
    
    return render_template("index.html", **index)
