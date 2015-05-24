#!/usr/bin/python

import os, glob

import flask, sqlalchemy
from flask import request, render_template, send_from_directory, current_app, session as current_session, jsonify
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

def buildPlateDesignDict(cubes):
    ''' Builds a list of dictionaries to pass to the plate design d3 code '''
    
    plateclass = cubes[0].plateclass

    #using xfocal,yfocal
    data = [{'name':plateclass.id,'cx':0.0,'cy':0.0,'r':200,'color':'white'}]
    for cube in cubes:
        hdr = cube.header_to_dict()
        data.append({'name':str(cube.ifu.name),'cx':cube.xfocal,'cy':cube.yfocal,'ra':float(hdr['OBJRA']),'dec':float(hdr['OBJDEC']),'r':5.0,'color':'red' if len(cube.ifu.name) > 3 else 'blue'})
        
    return data
    
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
    
    # Build rsync
    if not newtable:
        # No table data, just do it for a given plate
        
        # Grab all files and replace with SAS paths
        redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version,str(plate))
        sasredux = os.path.join(os.getenv('SAS_REDUX'),version,str(plate))    
    
        # Build the rsync path with the source and local paths
        dirpath = os.path.join(rsyncpath,sasredux,'stack')
        rsync_command = 'rsync -avz --progress --include "*{0}*fits*" {1} {2}'.format(id.upper(), dirpath, localpath)
    else:
        # table data from the search
        
        # grab versions from the table
        tablevers = newtable['versdrp3'].data
        tablevers = [v.split(' ')[0] if 'trunk' in v else v for v in tablevers]
        if len(set(tablevers)) == 1: version = tablevers[0]
        print('tablevers',tablevers)
        print('ver',version)
        
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
        
    return jsonify(result=rsync_command)

@plate_page.route('/marvin/plateInfo.html', methods=['GET'])
@plate_page.route('/plateInfo.html', methods=['GET'])
def plate():
    ''' Documentation here. '''
    
    session = db.Session() 
    plateinfo = {}
    plateinfo['title'] = "Marvin | Plate"
    
    # set global version
    try: version = current_session['currentver']
    except: 
        setGlobalVersion()
        version = current_session['currentver']
    plateinfo['version'] = version
    
    plate = valueFromRequest(key='plateID',request=request, default=None)
    plate = plate if plate.isdigit() else None
    plver = valueFromRequest(key='version',request=request, default=None)
    plateinfo['plate'] = plate
    plateinfo['inspection'] = None
    if plver: 
        version = plver
        plateinfo['version'] = plver
    
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
        commentdict=OrderedDict()
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
        inspection.set_version(version=version)
        inspection.set_ifudesign(plateid=plate)
        inspection.retrieve_cubecomments()
        inspection.retrieve_cubetags()
        inspection.retrieve_alltags()
        result = inspection.result()
        
        if inspection.ready: current_app.logger.warning('Inspection> GET cubecomments: {0}'.format(result))
        else: current_app.logger.warning('Inspection> NOT READY TO GET cubecomments: {0}'.format(result))

        # build plate design d3 
        jsdict={}
        jsdict['plateid'] = plate
        jsdict['platera'] = plateclass.ra if plateclass.cube else None
        jsdict['platedec'] = plateclass.dec if plateclass.cube else None
        jsdict['platedata'] =  buildPlateDesignDict(cubes) if plateclass.cube else None
        js = render_template('js/platedesign.js',**jsdict)
        plateinfo['js'] = js
        

    return render_template("plateInfo.html", **plateinfo)

