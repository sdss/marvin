#!/usr/bin/python

import os, glob, sys, traceback

import flask, sqlalchemy, json
from flask import request, redirect,render_template, send_from_directory, current_app, session as current_session, jsonify,url_for
from manga_utils import generalUtils as gu
from collections import OrderedDict
from ..model.database import db
from ..utilities import processTableData, getImages, setGlobalSession, parseError, getPlate3dDir, getDAPPlotDir
from comments import getComment
from astropy.table import Table
from sdss.manga import bundle
from flask_restful import Resource, reqparse, fields, marshal_with, abort 

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


def getMapsFiles(plate=None,version=None, table=None):
    ''' get the MAPS fits files for specified IFUs '''

    
    #mapsfile = os.path.join(os.getenv('SAS_URL'),'sas/mangawork/manga/sandbox/mangadap/MPL-4/default',str(plate),'mangadap-{0}-{1}-default.fits.gz'.format(plate,ifu))

    print('getmaps',plate,version,table)

    if not table and not all([plate,version]):
        raise RuntimeError('Plate and/or version not specified: {0},{1}'.format(plate,version))

    if plate and version:
        mapsdir = os.path.join(os.getenv('MANGA_SANDBOX'),'mangadap',version)
        if not os.path.isdir(mapsdir):
            raise RuntimeError('No MaNGA sandbox directory for {0}'.format(mapsdir))
        else:
            indirs = os.listdir(mapsdir)
            if len(indirs) > 1 or len(indirs) == 0:
                raise RuntimeError('Incorrect directories inside {0}: {1}'.format(mapsdir,indirs))
            else:  
                mapsdir = os.path.join(mapsdir,indirs[0],'default',str(plate))
                if not os.path.isdir(mapsdir):
                    raise RuntimeError('No MaNGA sandbox directory for {0}'.format(mapsdir))
                else:
                    mapsdir = os.path.join(mapsdir,'mangadap-{0}-*.fits*'.format(plate))

    # replace local sandbox with sas sandbox
    mapsplit = mapsdir.split('mangadap/',1)
    mapsdir = os.path.join('sas/mangawork/manga/sandbox','mangadap',mapsplit[1])

    return mapsdir
    
@plate_page.route('/downloadFiles', methods=['GET','POST'])
@plate_page.route('/marvin/downloadFiles', methods=['GET','POST'])
def downloadFiles():
    ''' Builds an rsync command to download all specified files '''

    result = {'message':None}    
    plate = valueFromRequest(key='plate',request=request, default=None)
    id = valueFromRequest(key='id',request=request, default=None)
    table = valueFromRequest(key='table',request=request, default=None)
    version = valueFromRequest(key='version',request=request, default=None)
    if not version:
        try: 
            version = current_session['currentver']
        except KeyError as e:
            result['message'] = 'KeyError getting rsync: {0}. Try refreshing your browser session'.format(e)
            result['status'] = -1
            return jsonify(result=result)

    print('stuff',plate,id,version)
    
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
        
        if not plate:
            result['status'] = -1
            result['message'] = 'No plate passed to Flask: {0}'.format(plate)
            return jsonify(result=result)

        # Grab all files and replace with SAS paths
        redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version,str(plate))
        sasredux = os.path.join(os.getenv('SAS_REDUX'),version,str(plate))    
    
        # Get MAPS or Cubes/RSS
        if id == 'maps':
            try: 
                mapfiles = getMapsFiles(plate,version=version)
            except RuntimeError as e:
                result['message'] = 'Error in getMapsFiles for {0},{1}: {2}'.format(plate,version,e)
                result['status'] = -1
                return jsonify(result=result) 

            # Make command
            dirpath = os.path.join(rsyncpath,mapfiles)
            rsync_command = 'rsync -avz --progress {0} {1}'.format(dirpath,localpath)
            result['command'] = rsync_command if rsync_command else None
            result['status'] = 1 if rsync_command else -1
            result['message'] = 'Success'
        else:
            # Build the rsync path with the source and local paths
            try: stagedir = getPlate3dDir(plate,version=version)
            except NameError as e:
                result['status'] = -1
                result['message'] = 'NameError in getPlate3dDir for {0}: {1}'.format(plate,e)
                return jsonify(result=result)
            except OSError as e:
                result['status'] = -1
                result['message'] = 'OSError in getPlate3dDir for {0}: {1}'.format(plate,e)
                return jsonify(result=result)            

            if stagedir:
                dirpath = os.path.join(rsyncpath,sasredux,stagedir)
                direxists = os.path.isdir(redux)
                if direxists:
                    if 'stack' in stagedir:
                        rsync_command = 'rsync -avz --progress --include "*{0}*fits*" {1} {2}'.format(id.upper(), dirpath, localpath)
                    elif 'mastar' in stagedir:
                        rsync_command = 'rsync -avz --progress --include "mastar*fits*" {0} {1}'.format(dirpath, localpath)
                    else:
                        rsync_command = None 
                    result['command'] = rsync_command if rsync_command else None
                    result['status'] = 1 if rsync_command else -1
                    result['message'] = 'mastar or stack not in output directory for {0}'.format(plate) if result['status'] == -1 else 'Success'
                else:
                    result['message'] = 'Directory path {0} does not exist'.format(sasredux) if not direxists else None
                    result['status'] = -1
            else:
                result['message'] = 'No 3d directory found for plate {0}'.format(plate)
                result['status'] = -1
    else:
        # table data from the search
        
        # grab versions from the table
        try: 
            tablevers = newtable['versdrp3'].data
            tablevers = [v.split(' ')[0] if 'trunk' in v else v for v in tablevers]
        except KeyError as e:
            tablevers = None
            result['message'] = 'Problem grabbing versions from table: KeyError {0}'.format(e)
            result['status'] = -1
            return jsonify(result=result)

        # Only grab 1 version
        if tablevers and len(set(tablevers)) == 1: version = tablevers[0]

        print('tablevers',tablevers)
        print('ver',version)  

        if tablevers:
            if id == 'maps':
                result['status']=-1
                result['message']='Feature not yet available'
            else:
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

def getBundle(ra, dec, size):
    ''' get bundle and hexagon coords of IFU for Aladin '''

    hexbundle = bundle.Bundle(ra, dec, size)
    coords = hexbundle.hexagon.tolist()
    coords.append(coords[0])

    return coords

def getDefaultMapsFile(plate,ifu):
    ''' temporary function to get the default Maps file, until DAP DB is in place '''

    mapsfile = os.path.join(os.getenv('SAS_URL'),'sas/mangawork/manga/sandbox/mangadap/MPL-4/default',str(plate),'mangadap-{0}-{1}-default.fits.gz'.format(plate,ifu))
    return mapsfile

def getDAPMapLink(plate,ifu):
    ''' temporary function to get the default DAP Maps Quick look html link '''

    dapredux = getDAPPlotDir(mode=False)
    dappiece = dapredux.split('analysis/')[1] 
    dapsas = os.path.join(os.getenv('SAS_URL'),os.getenv('SAS_ANALYSIS'),dappiece)
    pltgrp = '{0}00'.format(str(plate)[:-2])
    page = 'specind_{0}.html#{1}-{2}'.format(pltgrp,plate,ifu)
    daplink = os.path.join(dapsas,'full/html',page)
    return daplink

def getifu(plateid=None, ifuid=None, mangaid=None, version=None, dapversion=None):
    ''' get an ifu from the plate page '''

    session = db.Session()

    # query by plate and ifu id
    if all([plateid,ifuid]):
        try:
            cube = session.query(datadb.Cube).join(datadb.PipelineInfo,datadb.IFUDesign,
                datadb.PipelineVersion).filter(datadb.Cube.plate==int(plateid),datadb.PipelineVersion.version==version,datadb.IFUDesign.name==ifuid).one()
        except sqlalchemy.orm.exc.NoResultFound as error:
            raise RuntimeError('Error querying for single cube with plate {0}, ifuid {1}, and version {2}: {3}'.format(plateid,ifuid,version,error))

    # query by mangaid
    if mangaid:
        cubes = session.query(datadb.Cube).join(datadb.PipelineInfo,datadb.PipelineVersion).filter(datadb.Cube.mangaid==mangaid,datadb.PipelineVersion.version==version).all()
        if not cubes:
            raise RuntimeError('Error querying for mangaid {0}, version {1}.  Check the mangaid and/or version.'.format(mangaid,version))
        else:
            cube = cubes[0]
            plateid = cube.plate
            ifuid = cube.ifu.name

    # build ifu dictionary of parameter info
    try:
        images = getImages(plateid,version=version,ifuname=ifuid)
    except:
        type,val,trace = parseError(sys.exc_info())
        raise RuntimeError('Error retrieving image for plate {0}, ifuid {0}'.format(plateid, ifuid),type,val,trace)

    ifudict=OrderedDict()
    if cube:
        ifudict[cube.ifu.name]=OrderedDict()
        ifudict[cube.ifu.name]['image']=images[0] if images else None
        ifudict[cube.ifu.name]['sample']=OrderedDict()
        ifudict[cube.ifu.name]['mapsfile'] = getDefaultMapsFile(plateid, ifuid)
        ifudict[cube.ifu.name]['mapslink'] = getDAPMapLink(plateid, ifuid)
        hdr = json.loads(cube.hdr[0].header)         
        if cube.sample:
            try:
                for col in cube.sample[0].cols:
                    if ('absmag' in col) or ('flux' in col) or ('amivar' in col) or ('extinction' in col):
                        if 'absmag' in col: name='nsa_absmag_el' if 'el' in col else 'nsa_absmag'
                        if 'amivar' in col: name='nsa_amivar_el' if 'el' in col else 'nsa_amivar'
                        if 'extinction' in col: name='nsa_extinction_el' if 'el' in col else 'nsa_extinction'
                        if 'petro' in col: 
                            if 'el' in col: name='nsa_petroflux_el' if 'ivar' not in col else 'nsa_petroflux_el_ivar'
                            else: name='nsa_petroflux' if 'ivar' not in col else 'nsa_petroflux_ivar'
                        if 'sersic' in col: name='nsa_sersicflux' if 'ivar' not in col else 'nsa_sersicflux_ivar'
                        try: ifudict[cube.ifu.name]['sample'][name].append(cube.sample[0].__getattribute__(col))
                        except: ifudict[cube.ifu.name]['sample'][name] = [cube.sample[0].__getattribute__(col)]
                    elif 'ifu_' in col:
                        # NOTE: temporary solution only                      
                        if cube.sample[0].__getattribute__(col): ifudict[cube.ifu.name]['sample'][col] = cube.sample[0].__getattribute__(col)
                        else:
                            ifudict[cube.ifu.name]['sample'][col] = hdr[''.join(col.upper().split('_'))]
                    else:                         
                        ifudict[cube.ifu.name]['sample'][col] = cube.sample[0].__getattribute__(col)
            except:
                type,val,trace = parseError(sys.exc_info())
                raise RuntimeError('Error populating sample parameters for ifu {0}: '.format(ifuid),type,val,trace)

            # add hex bundle info
            ra = hdr['IFURA'] if version > 'v1_5_0' else hdr['OBJRA']
            dec = hdr['IFUDEC'] if version > 'v1_5_0' else hdr['OBJDEC']
            ifudict[cube.ifu.name]['target'] = '{0}, {1}'.format(ra,dec)
            ifudict[cube.ifu.name]['coords'] = getBundle(ra,dec,int(cube.ifu.name[:-2]))
            print('ifudict',ifudict[cube.ifu.name]['coords'])

    # get inspection information for ifu
    inspection=None
    if all([plateid,ifuid]):
        inspection = Inspection(current_session)
        if 'inspection_counter' in inspection.session: current_app.logger.info("Inspection Counter %r" % inspection.session['inspection_counter'])
        inspection.set_version(drpver=version,dapver=dapversion)
        inspection.set_ifudesign(plateid=plateid,ifuname=ifuid)
        try: 
            inspection.retrieve_cubecomments()
            current_app.logger.warning('Inspection> RETRIEVE cubecomments: {0}'.format(inspection.cubecomments))
            inspection.retrieve_dapqacubecomments()
            current_app.logger.warning('Inspection> RETRIEVE dapqacubecomments: {0}'.format(inspection.dapqacubecomments))
            inspection.retrieve_cubetags()
            inspection.retrieve_alltags()
        except:
            type,val,trace = parseError(sys.exc_info())
            raise RuntimeError('Error retrieving comments and tags from Inspection for plate {0},ifu {1}: '.format(plateid,ifuid),type,val,trace)

        result = inspection.result()
        if inspection.ready: current_app.logger.warning('Inspection> GET recentcomments: {0}'.format(result))
        else: current_app.logger.warning('Inspection> NOT READY TO GET recentcomments: {0}'.format(result)) 
    else:
        raise RuntimeError('Both plateid and ifuid must be specified as input to Inspection: plateid {0},ifuid {1}'.format(plateid,ifuid))

    return cube, ifudict, inspection


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

    # check if ifuid is actually a version 
    if ifuid and not ifuid.isdigit() and 'v' in ifuid:
        plver = ifuid
        ifuid = None

    plateinfo['plate'] = plateid
    cube=cubes= None
    if plver: 
        version = plver
        plateinfo['version'] = plver

    plateinfo['inspection'] = Inspection(current_session)

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
        # try to grab the ifu images
        try:
            images = getImages(plateid,version=version)
        except:
            type,val,trace = parseError(sys.exc_info())
            plateinfo['error'] = RuntimeError('Could not grab images for plate {0}: '.format(plateid),type,val,trace)
            return render_template('errors/no_plateid.html',**plateinfo)

        plateinfo['images'] = sorted(images) if images else None
        sasurl = os.path.join(sasurl,sasredux)
        plateinfo['sasurl'] = sasurl
        plateinfo['ifuid']  = ifuid

        # get all the cubes
        try:
            cubes = session.query(datadb.Cube).join(datadb.PipelineInfo,
                datadb.PipelineVersion).filter(datadb.Cube.plate==plateid,datadb.PipelineVersion.version==version).all()
            cubes=sorted(cubes,key=lambda t: t.ifu.name)
        except sqlalchemy.orm.exc.NoResultFound as error:
            plateinfo['error'] = 'Error querying for cubes on plate {0}, version {1}: {2}'.format(plateid,version,error)
            return render_template('errors/no_plateid.html',**plateinfo)

        # get the ifu information when necessary
        if ifuid:
            try:
                cube, ifudict, inspection = getifu(plateid=plateid, ifuid=ifuid, version=version, dapversion=dapversion)
            except RuntimeError as error:
                plateinfo['error'] = error
                return render_template('errors/no_plateid.html', **plateinfo)

            plateinfo['ifudict'] = ifudict
            plateinfo['inspection'] = inspection

        if not cube and cubes:
            cube = cubes[0]
        
        # get plate info and populate dict.
        plateclass = cube.plateclass if cube else datadb.Plate(id=plateid)
        plateinfo['plate'] = plateclass
        plateinfo['cube'] = cube

        # build plate design D3 
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

    ifu = get_mangaid(mangaid=mangaid,getver=getver, web=True)

    if type(ifu) != dict:
        return ifu

    return render_template('singleifu.html', **ifu)

def get_mangaid(mangaid=None,getver=None, ifu=None, web=None):

    if not ifu: ifu={}
    ifu['title'] = "Marvin | ID"
    ifu['mangaid'] = mangaid

    # set global session variables
    setGlobalSession()
    version = current_session['currentver']
    dapversion = current_session['currentdapver']    
    ifu['version'] = version
    ifu['dapversion'] = dapversion

    # reset new version if specified in url
    if getver: 
        version = getver
        ifu['version'] = getver

    # get the ifu information
    if mangaid:
        try:
            cube, ifudict, inspection = getifu(mangaid=mangaid,version=version, dapversion=dapversion)
        except RuntimeError as error:
            ifu['error'] = error
            if web:
                return render_template('errors/no_mangaid.html', **ifu)
            else:
                ifu['status'] = -1
                ifu['message'] = 'Could not complete request for mangaid {0}, due to indicated error.'.format(mangaid)
    else:
        if web: return render_template('errors/no_mangaid.html', **ifu)
        else:
            ifu['status'] = -1
            ifu['message'] = 'No mangaid specified'

    # push parameters to dict.
    if cube:
        ifu['plate'] = plate = cube.plate if cube else None
        ifu['name'] = ifuname = cube.ifu.name if cube else None
        ifu['cube'] = cube = cube if cube else None
        ifu['title'] += ' {0}'.format(mangaid)

    ifu['inspection'] = inspection 
    ifu['ifudict'] = ifudict

    return ifu

plates={}
class TestPlate(Resource):
    def get(self,plateid):
        if plateid in plates:
            return {plateid:plates[plateid]}
        else:
            return {plateid:'No plate found'}
    def post(self,plateid):
        plates[plateid] = request.form['data']
        return {plateid:plates[plateid]}

mangaids={'12-84660':'','12-98126':''}
ifu={}
resource_fields = {
    'task':fields.String,
    'mangaid':fields.Nested(ifu)
    }

def abort_if_todo_doesnt_exist(todo_id):
    if todo_id not in mangaids:
        abort(404, message="MangaID {} doesn't exist".format(todo_id))

class MangaIDList(Resource):
    def get(self):
        session=db.Session()
        mangaids = session.query(datadb.Cube.mangaid).all()
        mangaids = sorted(list(set([m[0] for m in mangaids])))
        return {'mangaids':mangaids}

class MangaID(Resource):
    #@marshal_with(resource_fields)
    def get(self,mangaid=None):

        parser = reqparse.RequestParser()
        parser.add_argument('mangaid', type=str, help='unique manga-id to retrieve')
        parser.add_argument('getver', type=str, help='version to retrieve')
        args = parser.parse_args(strict=True)
        print('args',args)

        abort_if_todo_doesnt_exist(mangaid)

        ifu = get_mangaid(mangaid=mangaid,getver=args.getver)

        if ifu['cube']:
            cols = ifu['cube'].cols
            ifu['cube'] = {col:ifu['cube'].__getattribute__(col) for col in cols}

        if ifu['inspection']:
            ifu['inspection'] = ifu['inspection'].result()

        return {'task':'hello',mangaid:ifu}       

    def post(self,mangaid):
        mangaids[mangaid] = request.form['data']
        return {mangaid:mangaids[mangaid]}        

