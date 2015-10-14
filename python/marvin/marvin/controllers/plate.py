#!/usr/bin/python

import os, glob

import flask, sqlalchemy, json
from flask import request, redirect,render_template, send_from_directory, current_app, session as current_session, jsonify,url_for
from manga_utils import generalUtils as gu
from collections import OrderedDict
from ..model.database import db
from ..utilities import processTableData, getImages, setGlobalSession
from comments import getComment
from astropy.table import Table

from flask_wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired 


import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try: from inspection.marvin import Inspection
except: from marvin.inspection import Inspection

try:
    from . import valueFromRequest
except ValueError:
    pass

plate_page = flask.Blueprint("plate_page", __name__)

def buildPlateDesignDict(cubes):
    ''' Builds a list of dictionaries to pass to the plate design d3 code '''
    
    plateclass = cubes[0].plateclass

    #using xfocal,yfocal
    data = [{'name':plateclass.id,'cx':0.0,'cy':0.0,'r':200,'color':'white'}]
    for cube in cubes:
        hdr = json.loads(cube.hdr[0].header)
        data.append({'name':str(cube.ifu.name),'cx':cube.xfocal,'cy':cube.yfocal,'ra':float(hdr['OBJRA']),'dec':float(hdr['OBJDEC']),'r':5.0,'color':'red' if len(cube.ifu.name) > 3 else 'blue'})
        
    return data

@plate_page.route('/navidselect',methods=['POST'])
@plate_page.route('/marvin/navidselect',methods=['POST'])
def navidselect():
    ''' Select plate id or manga id based from navigation bar '''


    plateid = valueFromRequest(key='plateid',request=request, default=None)
    mangaid = valueFromRequest(key='mangaid',request=request, default=None)
    #localhost = 'MANGA_LOCALHOST' in os.environ
    #sasvm = 'HOSTNAME' in os.environ and 'sas-vm' in os.environ['HOSTNAME']

    if plateid:
        current_session['searchmode']='plateid'
        return redirect(url_for('plate_page.plate',plateid=plateid)) #if sasvm else redirect(url_for('plate_page.plate',plateid=plateid,_external=True,_scheme='https'))

    if mangaid:
        current_session['searchmode']='mangaid'
        return redirect(url_for('plate_page.singleifu',mangaid=mangaid)) #if sasvm else redirect(url_for('plate_page.singleifu',mangaid=mangaid,_external=True,_scheme='https'))

    
@plate_page.route('/downloadFiles', methods=['POST'])
@plate_page.route('/marvin/downloadFiles', methods=['POST'])
def downloadFiles():
    ''' Builds an rsync command to download all specified files '''
    
    plate = valueFromRequest(key='plate',request=request, default=None)
    id = valueFromRequest(key='id',request=request, default=None)
    table = valueFromRequest(key='table',request=request, default=None)
    version = valueFromRequest(key='version',request=request, default=None)
    if not version: version = current_session['currentver']
    result = {'message':None}
    
    if table != 'null':
        try:
            newtable = processTableData(table) 
        except AttributeError as e:
            result['message'] = 'AttributeError getting rsync, in processTableData: {0}'.format(e)
            result['status'] = -1
            return jsonify(result=result)
        except KeyError as e:
            result['message'] = 'KeyError getting rsync, in processTableData: {0}'.format(e)
            result['status'] = -1
            return jsonify(result=result)
    else: newtable = None

    # Build some general paths
    rsyncpath = 'rsync://sdss@dtn01.sdss.utah.edu:/'
    localpath = '.'
    
    # Build rsync
    if not newtable:
        # No table data, just do it for a given plate
        
        # Grab all files and replace with SAS paths
        redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version,str(plate))
        sasredux = os.path.join(os.getenv('SAS_REDUX'),version,str(plate))    
    
        # Build the rsync path with the source and local paths
        dirpath = os.path.join(rsyncpath,sasredux,'stack')
        direxists = os.path.isdir(redux)
        result['message'] = 'Directory path {0} does not exist'.format(sasredux) if not direxists else None
        rsync_command = 'rsync -avz --progress --include "*{0}*fits*" {1} {2}'.format(id.upper(), dirpath, localpath)
        result['command'] = rsync_command if rsync_command else None
    else:
        # table data from the search
        
        # grab versions from the table
        try: 
            tablevers = newtable['versdrp3'].data
            tablevers = [v.split(' ')[0] if 'trunk' in v else v for v in tablevers]
        except KeyError as e:
            tablevers = None
            result['message'] = 'KeyError {0}'.format(e)

        if tablevers and len(set(tablevers)) == 1: version = tablevers[0]

        print('tablevers',tablevers)
        print('ver',version)  

        if tablevers:
            sasredux = os.path.join(os.getenv('SAS_REDUX'),version,'*')
            dirpath = os.path.join(rsyncpath,sasredux,'stack/')
        
            # handle individual files from table
            files = ['manga-{0}-{1}-LOG{2}.fits.gz'.format(row['plate'], row['ifudesign'],id.upper()) for row in newtable]
            includelist = [" --include='*{0}'".format(file) for file in files]
            includestring = ''.join(includelist)
        
            # build the string
            rsync_command = 'rsync -avz --prune-empty-dirs --progress'
            rsync_command += includestring 
            rsync_command += " --exclude='*' {0} {1}".format(dirpath,localpath)
            result['command'] = rsync_command if rsync_command else None

    return jsonify(result=result)


def getifu(cube=None):
    ''' get an ifu from the plate page '''

    plateid = valueFromRequest(key='plateid',request=request, default=None)
    ifuid = valueFromRequest(key='ifuid',request=request, default=None)
    session = db.Session()

    if not cube:
        cube = session.query(datadb.Cube).join(datadb.PipelineInfo,datadb.IFUDesign,
            datadb.PipelineVersion).filter(datadb.Cube.plate==int(plateid),datadb.PipelineVersion.version==version,datadb.IFUDesign.name==ifuid).one()

    ifudict=OrderedDict()
    imindex = [images.index(i) for i in images if '/{0}.png'.format(cube.ifu.name) in i]
    ifudict[cube.ifu.name]=OrderedDict()
    ifudict[cube.ifu.name]['image']=images[imindex[0]] if imindex else None
    if cube.sample:
        for col in cube.sample[0].cols:
            if ('absmag' in col) or ('flux' in col):
                if 'absmag' in col: name='nsa_absmag'
                if 'petro' in col: name='nsa_petroflux' if 'ivar' not in col else 'nsa_petroflux_ivar'
                if 'sersic' in col: name='nsa_sersicflux' if 'ivar' not in col else 'nsa_sersicflux_ivar'
                try: ifudict[cube.ifu.name][name].append(cube.sample[0].__getattribute__(col))
                except: ifudict[cube.ifu.name][name] = [cube.sample[0].__getattribute__(col)]
            else:
                ifudict[cube.ifu.name][col] = cube.sample[0].__getattribute__(col)
               

    result={}
    result['plate'] = plateid
    result['ifuid'] = ifuid
    result['ifudict'] = ifudict

    return jsonify(result=result)

@plate_page.route('/plate/')
@plate_page.route('/plate/<int:plateid>/')
@plate_page.route('/plate/<int:plateid>/<plver>') 
@plate_page.route('/plate/<int:plateid>/<ifuid>/')
@plate_page.route('/plate/<int:plateid>/<ifuid>/<plver>/')
def plate(plateid=None, plver=None, ifuid=None):
    session = db.Session() 
    plateinfo = {}
    plateinfo['title'] = "Marvin | Plate"
    
    # set global session variables
    setGlobalSession()
    version = current_session['currentver']
    dapversion = current_session['currentdapver'] 
    plateinfo['version'] = version
    plateinfo['dapversion'] = dapversion

    # Check if input is plateID for mangaID    
    #plate = valueFromRequest(key='plateid',request=request, default=None)
    #plate = plate if plate and plate.isdigit() else None

    print('early input ifu',plateid,ifuid)

    # check if ifuid is actually a version 
    if ifuid and not ifuid.isdigit() and 'v' in ifuid:
        plver = ifuid
        ifuid = None

    #plver = valueFromRequest(key='version',request=request, default=None)
    plateinfo['plate'] = plateid
    plateinfo['inspection'] = None
    cube=cubes= None
    if plver: 
        version = plver
        plateinfo['version'] = plver

    plateinfo['inspection'] = inspection = Inspection(current_session)

    # Get info from plate   
    if plateid:
        plateinfo['title'] += " {0}".format(plateid)
        # find and grab all images for plate ; convert paths to sas paths
        redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version,str(plateid))
        good = os.path.isdir(redux)
        plateinfo['good'] = good
        sasredux = os.path.join(os.getenv('SAS_REDUX'),version,str(plateid))
        try: sasurl = os.getenv('SAS_URL')
        except: sasurl= None
        images = getImages(plateid,version=version)
        plateinfo['images'] = sorted(images) if images else None
        print('sas stuff',sasredux, sasurl)
        sasurl = os.path.join(sasurl,sasredux)
        plateinfo['sasurl'] = sasurl
        
        # get cubes for this plate and current tag, sort by ifu name
        if ifuid:
            try:
                cube = session.query(datadb.Cube).join(datadb.PipelineInfo,datadb.IFUDesign,
                    datadb.PipelineVersion).filter(datadb.Cube.plate==plateid,datadb.PipelineVersion.version==version,datadb.IFUDesign.name==ifuid).one()
            except sqlalchemy.orm.exc.NoResultFound as error:
                plateinfo['error'] = 'Error querying for single cube with plate {0}, ifuid {1}, and version {2}: {3}'.format(plateid,ifuid,version,error)
                plateinfo['ifuid']  = ifuid
                return render_template('errors/no_plateid.html',**plateinfo)

        # get all the cubes
        try:
            cubes = session.query(datadb.Cube).join(datadb.PipelineInfo,
                datadb.PipelineVersion).filter(datadb.Cube.plate==plateid,datadb.PipelineVersion.version==version).all()
            cubes=sorted(cubes,key=lambda t: t.ifu.name)
        except sqlalchemy.orm.exc.NoResultFound as error:
            plateinfo['error'] = error
            return render_template('errors/no_plateid.html',**plateinfo)

        if not cube and cubes:
            cube = cubes[0]
        
        # get plate info
        plateclass = cube.plateclass if cube else datadb.Plate(id=plateid)
        plateinfo['plate'] = plateclass

        # ifu
        ifudict=OrderedDict()
        if ifuid:
            #cube = [cube for cube in cubes if cube.ifu.name == ifuid][0]
            imindex = [images.index(i) for i in images if '/{0}.png'.format(cube.ifu.name) in i]
            ifudict[cube.ifu.name]=OrderedDict()
            ifudict[cube.ifu.name]['image']=images[imindex[0]] if imindex else None
            if cube.sample:
                for col in cube.sample[0].cols:
                    if ('absmag' in col) or ('flux' in col):
                        if 'absmag' in col: name='nsa_absmag'
                        if 'petro' in col: name='nsa_petroflux' if 'ivar' not in col else 'nsa_petroflux_ivar'
                        if 'sersic' in col: name='nsa_sersicflux' if 'ivar' not in col else 'nsa_sersicflux_ivar'
                        try: ifudict[cube.ifu.name][name].append(cube.sample[0].__getattribute__(col))
                        except: ifudict[cube.ifu.name][name] = [cube.sample[0].__getattribute__(col)]
                    else:
                        ifudict[cube.ifu.name][col] = cube.sample[0].__getattribute__(col)
                       
        plateinfo['cube'] = cube
        plateinfo['ifuid'] = ifuid
        plateinfo['ifudict'] = ifudict

        # set comment information
        if 'http_authorization' not in current_session:
            try: current_session['http_authorization'] = request.environ['HTTP_AUTHORIZATION']
            except: pass
        
        plateinfo['inspection'] = inspection = Inspection(current_session)
        if 'inspection_counter' in inspection.session: current_app.logger.info("Inspection Counter %r" % inspection.session['inspection_counter'])
        inspection.set_version(drpver=version,dapver=dapversion)

        print('pre input ifu',plateid,ifuid)

        if ifuid:
            print('input ifu',plateid,ifuid)
            inspection.set_ifudesign(plateid=plateid,ifuname=ifuid)
            print('inspection ifu',inspection.ifudesign)
            inspection.retrieve_cubecomments()
            current_app.logger.warning('Inspection> RETRIEVE cubecomments: {0}'.format(inspection.cubecomments))
            inspection.retrieve_dapqacubecomments()
            current_app.logger.warning('Inspection> RETRIEVE dapqacubecomments: {0}'.format(inspection.dapqacubecomments))
            inspection.retrieve_cubetags()
            inspection.retrieve_alltags()

        result = inspection.result()        
        if inspection.ready: current_app.logger.warning('Inspection> GET recentcomments: {0}'.format(result))
        else: current_app.logger.warning('Inspection> NOT READY TO GET recentcomments: {0}'.format(result))

        # build plate design d3 
        jsdict={}
        jsdict['plateid'] = plateid
        jsdict['platera'] = plateclass.ra if plateclass.cube else None
        jsdict['platedec'] = plateclass.dec if plateclass.cube else None
        jsdict['platedata'] =  buildPlateDesignDict(cubes) if plateclass.cube else None
        js = render_template('js/platedesign.js',**jsdict)
        plateinfo['platedesignjs'] = js

    return render_template("plateInfo.html", **plateinfo)    

@plate_page.route('/mangaid/')
@plate_page.route('/mangaid/<mangaid>/')
@plate_page.route('/mangaid/<mangaid>/<getver>/')
def singleifu(mangaid=None, getver=None):
    ''' '''

    session = db.Session()
    ifu={}
    ifu['title'] = "Marvin | ID"
    
    # set global session variables
    setGlobalSession()
    version = current_session['currentver']
    dapversion = current_session['currentdapver']    
    ifu['version'] = version
    ifu['dapversion'] = dapversion

    #mangaid = valueFromRequest(key='mangaid',request=request, default=None)
    ifu['mangaid'] = mangaid
    #getver = valueFromRequest(key='version',request=request, default=None)
    if getver: 
        version = getver
        ifu['version'] = getver

    if mangaid:
        cube = session.query(datadb.Cube).join(datadb.PipelineInfo,datadb.PipelineVersion).filter(datadb.Cube.mangaid==mangaid,datadb.PipelineVersion.version==version).all()
        ifu['plate'] = plate = cube[0].plate if cube else None
        ifu['name'] = ifuname = cube[0].ifu.name if cube else None
        ifu['cube'] = cube = cube[0] if cube else None
        ifu['title'] += ' {0}'.format(mangaid)
    else:
        return render_template('errors/no_mangaid.html', **ifu)

    # image
    images = getImages(plate,version=version,ifuname=ifuname)
    ifu['images'] = sorted(images) if images else None

    # Inspection
    if 'http_authorization' not in current_session:
        try: current_session['http_authorization'] = request.environ['HTTP_AUTHORIZATION']
        except: pass
        
    ifu['inspection'] = inspection = Inspection(current_session)
    if 'inspection_counter' in inspection.session: current_app.logger.info("Inspection Counter %r" % inspection.session['inspection_counter'])
    inspection.set_version(drpver=version,dapver=dapversion)
    inspection.set_ifudesign(plateid=plate,ifuname=ifuname)
    inspection.retrieve_cubecomments()
    current_app.logger.warning('Inspection> RETRIEVE cubecomments: {0}'.format(inspection.cubecomments))
    inspection.retrieve_dapqacubecomments()
    current_app.logger.warning('Inspection> RETRIEVE dapqacubecomments: {0}'.format(inspection.dapqacubecomments))
    inspection.retrieve_cubetags()
    inspection.retrieve_alltags()
    result = inspection.result()

    if inspection.ready: current_app.logger.warning('Inspection> GET recentcomments: {0}'.format(result))
    else: current_app.logger.warning('Inspection> NOT READY TO GET recentcomments: {0}'.format(result))

    # put sample info into a dictionary
    ifudict=OrderedDict()
    if cube:
        ifudict[cube.ifu.name]=OrderedDict()
        ifudict[cube.ifu.name]['image']=images[0] 
        if cube.sample:
            for col in cube.sample[0].cols:
                if ('absmag' in col) or ('flux' in col):
                    if 'absmag' in col: name='nsa_absmag'
                    if 'petro' in col: name='nsa_petroflux' if 'ivar' not in col else 'nsa_petroflux_ivar'
                    if 'sersic' in col: name='nsa_sersicflux' if 'ivar' not in col else 'nsa_sersicflux_ivar'
                    try: ifudict[cube.ifu.name][name].append(cube.sample[0].__getattribute__(col))
                    except: ifudict[cube.ifu.name][name] = [cube.sample[0].__getattribute__(col)]
                    else:
                        ifudict[cube.ifu.name][col] = cube.sample[0].__getattribute__(col)
    ifu['ifudict'] = ifudict

    return render_template('singleifu.html', **ifu)



