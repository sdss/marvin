#!/usr/bin/python

import os, glob

import flask, sqlalchemy
from flask import request, render_template, send_from_directory, current_app, session as current_session, jsonify
from manga_utils import generalUtils as gu
from collections import OrderedDict
from ..model.database import db
from ..utilities import processTableData
from comments import getComment
from astropy.table import Table

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try: from inspection.manga import Feedback
except: from mangasas.inspection import Feedback

try:
    from . import valueFromRequest
except ValueError:
    pass

plate_page = flask.Blueprint("plate_page", __name__)

def buildPlateDesignDict(cubes):
    ''' Builds a list of dictionaries to pass to the plate design d3 code '''
    
    plateclass = cubes[0].plateclass
    #data = [{'cx':float(plateclass.ra),'cy':float(plateclass.dec),'r':200,'color':'white'}]
    #data.extend([{'name':str(cube.ifu.name),'cx':float(cube.header['OBJRA']),'cy':float(cube.header['OBJDEC']),'r':5.0,'color':'red' if len(cube.ifu.name) > 3 else 'blue'} for cube in cubes])
    
    #using xfocal,yfocal
    data = [{'name':plateclass.id,'cx':0.0,'cy':0.0,'r':200,'color':'white'}]
    data.extend([{'name':str(cube.ifu.name),'cx':cube.xfocal,'cy':cube.yfocal,'ra':float(cube.header_to_dict()['OBJRA']),'dec':float(cube.header_to_dict()['OBJDEC']),'r':5.0,'color':'red' if len(cube.ifu.name) > 3 else 'blue'} for cube in cubes])

    return data

@plate_page.route('/manga/downloadFiles', methods=['GET','POST'])
@plate_page.route('/downloadFiles', methods=['GET','POST'])
def downloadFiles():
    ''' Builds an rsync command to download all specified files '''
    
    plate = valueFromRequest(key='plate',request=request, default=None)
    version = valueFromRequest(key='version',request=request, default=gu.getMangaVersion(simple=True))    
    id = valueFromRequest(key='id',request=request, default=None)
    table = valueFromRequest(key='table',request=request, default=None)
    
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

@plate_page.route('/plateInfo.html', methods=['GET'])
def plate():
    ''' Documentation here. '''
    
    session = db.Session() 
    plateinfo = {}
    
    plate = valueFromRequest(key='plateID',request=request, default=None)
    version = valueFromRequest(key='version',request=request, default=None)
    plateinfo['plate'] = plate
    
    if plate:
        # define version if none 
        if not version: version = gu.getMangaVersion(simple=True)
        plateinfo['version'] = str(version)
        
        # find and grab all images for plate ; convert paths to sas paths
        redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version,str(plate))
        sasredux = os.path.join(os.getenv('SAS_REDUX'),version,str(plate))
        good = os.path.isdir(redux)
        plateinfo['good'] = good
        imagedir = os.path.join(redux,'stack','images')
        images = glob.glob(os.path.join(imagedir,'*.png'))
        images = [os.path.join(os.getenv('SAS_URL'),sasredux,'stack/images',i.split('/')[-1]) for i in images]
        plateinfo['images'] = sorted(images) if images else None
        sasurl = os.path.join(os.getenv('SAS_URL'),sasredux)
        plateinfo['sasurl'] = os.path.join(os.getenv('SAS_URL'),sasredux)
        
        # get cubes for this plate and current tag
        cubes = session.query(datadb.Cube).join(datadb.PipelineInfo,
            datadb.PipelineVersion).filter(datadb.Cube.plate==plate,datadb.PipelineVersion.version==version).all()
        
        # get plate info
        plateclass = cubes[0].plateclass if cubes else datadb.Plate(id=plate)
        plateinfo['plate'] = plateclass
         
        # put sample info into a dictionary
        ifudict=OrderedDict()
        commentdict=OrderedDict()
        for cube in cubes:
            imindex = [images.index(i) for i in images if cube.ifu.name in i]
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
                        
            # get comments for each cube and store in comment dictionary (commentdict)
            #comments = getComments(all=True, cube=cube)
            #commentdict[cube.ifu.name] = Table({'comments':comments})
        
        plateinfo['cubes'] = cubes
        plateinfo['ifudict'] = ifudict
        
        # set comment information
        plateinfo['feedback'] = Feedback(current_session)
        if 'feedback_counter' in plateinfo['feedback'].session: current_app.logger.info("Feedback counter %r" % plateinfo['feedback'].session['feedback_counter'])
        plateinfo['comments'] = None #commentdict
        
        # build plate design d3 
        jsdict={}
        jsdict['plateid'] = plate
        jsdict['platera'] = plateclass.ra if plateclass.cube else None
        jsdict['platedec'] = plateclass.dec if plateclass.cube else None
        jsdict['platedata'] =  buildPlateDesignDict(cubes) if plateclass.cube else None
        js = render_template('js/platedesign.js',**jsdict)
        plateinfo['js'] = js
        

    return render_template("plateInfo.html", **plateinfo)

