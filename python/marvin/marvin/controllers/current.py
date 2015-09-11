#!/usr/bin/python

import os, glob
import flask, sqlalchemy
from flask import request, render_template, send_from_directory, current_app, jsonify
from flask import session as current_session
from ..model.database import db
import sdss.internal.database.utah.mangadb.DataModelClasses as datadb
import datetime, fitsio
from manga_utils import generalUtils as gu
from runmanga import plateList, setStatusDone
from collections import OrderedDict

from ..utilities import makeQualNames, getMaskBitLabel, setGlobalVersion
try:
    from . import valueFromRequest
except ValueError:
    pass 



def getStatus(path, stage='2d'):
    ''' Get new reduction status '''
    
    subdir = 'stack' if stage=='3d' else '*'
    mangadir = os.path.join(path,'*/{0}/manga{1}*.*'.format(subdir,stage))
    files = glob.glob(mangadir)
    
    # get status
    status = ['NULL','err','out','transferred','queued', 'started', 'running', 'done', 'fault', 'redo']    
    plates = list(set([s.split('-')[-1].split('.')[0] for s in files] ))
    stats=[[f.split('.')[-1] for f in files if str(plate) in f] for plate in plates]
    finalstat = [status[max([status.index(ind) for ind in stat])] for stat in stats]
    
    # get plates, mjds, and cubes
    plates = [p for p in os.listdir(path) if p.isdigit()]
    
    if stage == '2d':
        grp = [[p,mjd] for p in plates for mjd in os.listdir(os.path.join(path,p)) if mjd.isdigit()]
        mjds = list(set([long(item[1]) for item in grp]))
    else:
        grp = [[p,mjd] for p in plates for mjd in os.listdir(os.path.join(path,p)) if mjd == 'stack']
        plates = list(set([long(item[0]) for item in grp]))
        mjds = [glob.glob(os.path.join(path,item[0],item[1],'manga-*LOGCUBE*.*')) for item in grp]
        fullcube = []
        tmp=map(fullcube.extend, [f for f in mjds if f != []])
        mjds = fullcube
        
    return finalstat, plates, mjds
    
"""
def makeQualNames(bits, stage='2d'):
    ''' Return list containing the quality flag names '''
     
    name = 'MANGA_DRP2QUAL' if stage=='2d' else 'MANGA_DRP3QUAL'
    flagnames = [gu.getSDSSFlagName(bit,name=name) for bit in bits]
    return flagnames
"""    
def buildPlateDict(stage='2d', cols=None):
    ''' Build the platelist into dictionary form '''
    
    # load platelist
    pl=plateList()
    pl.load()
    table = pl.plate2d if stage == '2d' else pl.plate3d
    
    # select unique plates    
    uniqplates = list(set(table['plate']))
    
    # build dictionary by key,val = plate: row
    plateDict = {plate:{col:list(table[col][table['plate']==plate]) for col in cols} for plate in uniqplates}
    for plate in uniqplates: plateDict[plate]['count'] = len(plateDict[plate]['plate'])
    
    return OrderedDict(sorted(plateDict.iteritems()))    

current_page = flask.Blueprint("current_page", __name__)

@current_page.route('/marvin/current.html', methods=['GET','POST'])    
@current_page.route('/current.html', methods=['GET','POST'])
def current():
    ''' Documentation here. '''
    
    session = db.Session()
    current = {}
    current['title'] = "Marvin | Current Reduction Status"
    jsDict={}
    
    # platelist type
    plate2d = valueFromRequest(key='plate2d',request=request, default=None)
    plate3d = valueFromRequest(key='plate3d',request=request, default=None)
    sortid = valueFromRequest(key='sortid', request=request, default=None)
    stage = '2d' if plate2d else '3d' if plate3d else None
    current['stage'] = stage
        
    # redux version
    try: version = current_session['currentver']
    except: 
        setGlobalVersion()
        version = current_session['currentver']
         
    # platelist and drpall files
    redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version)
    sasredux = os.path.join(os.getenv('SAS_REDUX'),version)
    plfile = os.path.join(os.getenv('SAS_URL'),sasredux, 'platelist.fits')
    drpall = os.path.join(os.getenv('SAS_URL'),sasredux, 'drpall-{0}.fits'.format(version))
    current['plfile'] = plfile
    current['drpall'] = drpall

    # redux status
    status = ['done','running', 'started', 'queued', 'redo', 'fault', 'transferred','NULL']
    stats2d, plates, mjds = getStatus(redux,stage='2d')    
    stats3d, plates3d, cubes = getStatus(redux,stage='3d')
    current['stats2d'] = stats2d
    current['stats3d'] = stats3d
    current['status'] = status
    current['plates'] = plates
    current['mjds'] = mjds
    current['plates3d'] = plates3d
    current['cubes'] = cubes
    current['faults'] = None
    
    # set display columns
    if stage == '2d':
        displayCols = ['plate','mjd','apocomp','versdrp2','status2d','type','complete','drp2qual','badifu','cartid','platetyp','srvymode','nexp','nscigood','nscibad','totalexptime','b1sn2','r1sn2','b2sn2','r2sn2']
    elif stage == '3d':
        displayCols = ['plate','ifudsgn','harname','mangaid','status3d','drp3qual','versdrp3','verscore','mjdmax','objRA', 'objDEC','bluesn2','redsn2','nexp','exptime']
        
    # Load faults
    try: 
        sd = setStatusDone(outver=version,nowrite=True)
    except TypeError as e:
        sd = None
        current_app.logger.error('Error accessing platelist.setStatusDone. TypeError: {0}'.format(e))
    if sd: sd.loadFile()
    errs = sd.getErrorDict() if sd else None
    current['faults'] = errs if errs else None
    	
    # Make platelist
    if stage:  
        pl = plateList(outver=version)
        pl.load()	  
        table = pl.plate2d if stage == '2d' else pl.plate3d
        current['cols'] = displayCols
        current['platelist'] = table
        if table:
            current['uniqplates'] = sorted(list(set(table['plate'])))
            #flags = makeQualNames(table['drp2qual'],stage='2d') if stage == '2d' else makeQualNames(table['drp3qual'],stage='3d')
            flags = getMaskBitLabel(table['drp2qual']) if stage == '2d' else getMaskBitLabel(table['drp3qual'])
            current['flags']=flags
            keys = displayCols[:]
            keys.extend([x for x in table.keys() if x not in displayCols])
            current['keys'] = keys
    
    return render_template("current.html", **current)

