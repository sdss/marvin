#!/usr/bin/python

import os, glob

import flask, sqlalchemy
from flask import request, redirect,render_template, send_from_directory, current_app, session as current_session, jsonify,url_for
from manga_utils import generalUtils as gu
from collections import OrderedDict
from ..model.database import db
from ..utilities import processTableData, setGlobalVersion, getImages
from comments import getComment
from astropy.table import Table

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try: from inspection.marvin import Inspection
except: from marvin.inspection import Inspection

try:
    from . import valueFromRequest
except ValueError:
    pass

plate_page = flask.Blueprint("plate_page", __name__)


def setupInspection():
    ''' set up the inspection system '''

    return ''

def buildPlateDesignDict(cubes):
    ''' Builds a list of dictionaries to pass to the plate design d3 code '''
    
    plateclass = cubes[0].plateclass

    #using xfocal,yfocal
    data = [{'name':plateclass.id,'cx':0.0,'cy':0.0,'r':200,'color':'white'}]
    for cube in cubes:
        hdr = cube.header_to_dict()
        data.append({'name':str(cube.ifu.name),'cx':cube.xfocal,'cy':cube.yfocal,'ra':float(hdr['OBJRA']),'dec':float(hdr['OBJDEC']),'r':5.0,'color':'red' if len(cube.ifu.name) > 3 else 'blue'})
        
    return data

@plate_page.route('/marvin/navidselect',methods=['GET'])
@plate_page.route('/navidselect',methods=['GET'])
def navidselect():
    ''' Select plate id or manga id based from navigation bar '''

    plateid = valueFromRequest(key='plateid',request=request, default=None)
    mangaid = valueFromRequest(key='mangaid',request=request, default=None)
    localhost = 'MANGA_LOCALHOST' in os.environ
    utah = 'UUFSCELL' in os.environ['UUFSCELL'] and os.environ['UUFSCELL'] == 'kingspeak.peaks'

    if plateid:
        return redirect(url_for('plate_page.plate',plateid=plateid)) if utah else redirect(url_for('plate_page.plate',plateid=plateid,_external=True,_scheme='https'))

    if mangaid:
        return redirect(url_for('plate_page.singleifu',mangaid=mangaid)) if utah else redirect(url_for('plate_page.singleifu',mangaid=mangaid,_external=True,_scheme='https'))

    
@plate_page.route('/marvin/downloadFiles', methods=['GET','POST'])
@plate_page.route('/downloadFiles', methods=['GET','POST'])
def downloadFiles():
    ''' Builds an rsync command to download all specified files '''
    
    plate = valueFromRequest(key='plate',request=request, default=None)
    id = valueFromRequest(key='id',request=request, default=None)
    table = valueFromRequest(key='table',request=request, default=None)
    version = valueFromRequest(key='version',request=request, default=None)
    if not version: version = current_session['currentver']
    newtable = processTableData(table) if table != 'null' else None
    
    # Build some general paths
    rsyncpath = 'rsync://sdss@dtn01.sdss.utah.edu:/'
    localpath = '.'
    result = {'message':None}
    
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

@plate_page.route('/marvin/plateInfo.html', methods=['GET'])
@plate_page.route('/plateInfo.html', methods=['GET'])
def plate():
    ''' Documentation here. '''
    
    session = db.Session() 
    plateinfo = {}
    plateinfo['title'] = "Marvin | Plate"
    
    # set global version
    try: 
        version = current_session['currentver']
        dapversion = current_session['currentdapver']
    except: 
        setGlobalVersion()
        version = current_session['currentver']
        dapversion = current_session['currentdapver']
    plateinfo['version'] = version
    plateinfo['dapversion'] = dapversion

    # Check if input is plateID for mangaID    
    plate = valueFromRequest(key='plateid',request=request, default=None)
    plate = plate if plate.isdigit() else None 

    plver = valueFromRequest(key='version',request=request, default=None)
    plateinfo['plate'] = plate
    plateinfo['inspection'] = None
    if plver: 
        version = plver
        plateinfo['version'] = plver

    # Get info from plate   
    if plate:
        plateinfo['title'] += " {0}".format(plate)
        # find and grab all images for plate ; convert paths to sas paths
        redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version,str(plate))
        good = os.path.isdir(redux)
        plateinfo['good'] = good
        sasredux = os.path.join(os.getenv('SAS_REDUX'),version,str(plate))
        try: sasurl = os.getenv('SAS_URL')
        except: sasurl= None
        images = getImages(plate,version=version)
        plateinfo['images'] = sorted(images) if images else None
        sasurl = os.path.join(sasurl,sasredux)
        plateinfo['sasurl'] = os.path.join(sasurl,sasredux)
        
        # get cubes for this plate and current tag, sort by ifu name
        cubes = session.query(datadb.Cube).join(datadb.PipelineInfo,
            datadb.PipelineVersion).filter(datadb.Cube.plate==plate,datadb.PipelineVersion.version==version).all()
        cubes=sorted(cubes,key=lambda t: t.ifu.name)
        
        # get plate info
        plateclass = cubes[0].plateclass if cubes else datadb.Plate(id=plate)
        plateinfo['plate'] = plateclass
         
        # put sample info into a dictionary
        ifudict=OrderedDict()
        for cube in cubes:
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
                        
        
        plateinfo['cubes'] = cubes
        plateinfo['ifudict'] = ifudict
        
        # set comment information
        if 'http_authorization' not in current_session:
            try: current_session['http_authorization'] = request.environ['HTTP_AUTHORIZATION']
            except: pass
        
        plateinfo['inspection'] = inspection = Inspection(current_session)
        if 'inspection_counter' in inspection.session: current_app.logger.info("Inspection Counter %r" % inspection.session['inspection_counter'])
        inspection.set_version(drpver=version,dapver=dapversion)
        inspection.set_ifudesign(plateid=plate)
        inspection.retrieve_cubecomments()
        current_app.logger.warning('Inspection> RETRIEVE cubecomments: {0}'.format(inspection.cubecomments))
        inspection.retrieve_dapqacubecomments()
        current_app.logger.warning('Inspection> RETRIEVE dapqacubecomments: {0}'.format(inspection.dapqacubecomments))
        inspection.retrieve_cubetags()
        inspection.retrieve_alltags()
        result = inspection.result()
        print('result',result)
        
        if inspection.ready: current_app.logger.warning('Inspection> GET recentcomments: {0}'.format(result))
        else: current_app.logger.warning('Inspection> NOT READY TO GET recentcomments: {0}'.format(result))

        # build plate design d3 
        jsdict={}
        jsdict['plateid'] = plate
        jsdict['platera'] = plateclass.ra if plateclass.cube else None
        jsdict['platedec'] = plateclass.dec if plateclass.cube else None
        jsdict['platedata'] =  buildPlateDesignDict(cubes) if plateclass.cube else None
        js = render_template('js/platedesign.js',**jsdict)
        plateinfo['js'] = js
        

    return render_template("plateInfo.html", **plateinfo)

@plate_page.route('/marvin/singleifu', methods=['GET'])
@plate_page.route('/singleifu', methods=['GET'])
def singleifu():
    ''' '''

    session = db.Session()
    ifu={}
    ifu['title'] = "Marvin | ID"
    
    # set global version
    try: 
        version = current_session['currentver']
        dapversion = current_session['currentdapver']
    except: 
        setGlobalVersion()
        version = current_session['currentver']
        dapversion = current_session['currentdapver']
    ifu['version'] = version
    ifu['dapversion'] = dapversion

    mangaid = valueFromRequest(key='mangaid',request=request, default=None)
    ifu['mangaid'] = mangaid
    getver = valueFromRequest(key='version',request=request, default=None)
    if getver: 
        version = getver
        ifu['version'] = getver

    if mangaid:
        cube = session.query(datadb.Cube).join(datadb.PipelineInfo,datadb.PipelineVersion).filter(datadb.Cube.mangaid==mangaid,datadb.PipelineVersion.version==version).all()
        ifu['plate'] = plate = cube[0].plate if cube else None
        ifu['name'] = ifuname = cube[0].ifu.name if cube else None
        ifu['cube'] = cube = cube[0] if cube else None
        ifu['title'] += ' {0}'.format(mangaid)

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
    inspection.set_ifudesign(plateid=plate)
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


