#!/usr/bin/python

import os

import flask, sqlalchemy, json, tempfile
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import text
from flask import request, render_template, send_from_directory, current_app, jsonify, Response
from flask import session as current_session
from manga_utils import generalUtils as gu
from astropy.table import Table
from astropy.io import fits
from collections import defaultdict
import numpy as np

from ..model.database import db
from ..utilities import makeQualNames, processTableData

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try: from inspection.manga import Feedback
except: from mangasas.inspection import Feedback

try:
    from . import valueFromRequest
except ValueError:
    pass 


def doRawSQLSearch(sqlstring):
    ''' Do a raw sql search with the input from the Direct SQL search page '''
    
    sql = text(sqlstring)
    
    # Do the search
    try:
        result = db.engine.execute(sql)
    except sqlalchemy.exc.ProgrammingError:
        result = None
    
    return result

def buildSQLTable(cubes):
    ''' Build a table from the output of the direct SQL search via dictionaries '''
    
    cols = cubes[0].keys()
    displayCols = [col for col in cols if col not in ['specres','wavelength','flux','ivar','mask']]
    displayCols = cols if len(displayCols) <= 14 else displayCols[0:14] 
    
    # make list of dictionaries
    dictlist = [dict(zip(cube.keys(),cube)) for cube in cubes]
    cubedict=defaultdict(list)
    
    # restructure into dictionary of lists
    for d in dictlist:
        for key,val in d.items():
            cubedict[key].append(val)
    cubedict = dict(cubedict.items())
    
    # convert to Table
    cubetable = Table(cubedict)[cols]
    
    return cubetable, displayCols

def buildTable(cubes):
    ''' Build the data table as a dictionary '''
    
    cols = ['plate','ifudesign','mangaid','versdrp2','versdrp3','verscore','versutil','platetyp','srvymode',
    'objra','objdec','objglon','objglat','ebvgal','nexp','exptime','drp3qual','bluesn2','redsn2','harname','frlplug',
    'cartid','designid','cenra','cendec','airmsmin','airmsmed','airmsmax','seemin','seemed','seemax','transmin',
    'transmed','transmax','mjdmin','mjdmed','mjdmax','ufwhm','gfwhm','rfwhm','ifwhm','zfwhm','mngtarg1','mngtarg2',
    'mngtarg3','catidnum','plttarg']
    
    displayCols=['plate','ifudesign','harname','mangaid','drp3qual','versdrp3','verscore','mjdmax','objra', 'objdec','bluesn2','redsn2','nexp','exptime']
    
    cubedict=defaultdict(list)
    for cube in cubes:
        for col in cols:
            if col=='plate': cubedict['plate'].append(cube.plate)
            elif col=='mangaid': cubedict['mangaid'].append(cube.mangaid)
            elif col=='designid': cubedict['designid'].append(cube.designid)
            elif col=='ifudesign': cubedict['ifudesign'].append(cube.ifu.name)
            elif 'mngtarg' in col: 
                try: cubedict[col].append(cube.header_to_dict()[''.join(col.split('a'))])
                except: cubedict[col].append(None) 
            else: 
                try: cubedict[col].append(cube.header_to_dict()[col])
                except: cubedict[col].append(None)
                
    cubetable = Table(cubedict)
    cubetable = cubetable[cols]
    
    return cubetable, displayCols
    

def buildSQLString(minplate=None, maxplate=None, minmjd=None, maxmjd=None, 
                tag=gu.getMangaVersion(simple=True),type='any',ifu='any', sql=None,
                user=None, keyword=None, date=None, cat='any', issues='any'):
    ''' Build a string version of the SQLalchemy query'''
    
    query = 'session.query(datadb.Cube)'

    # Plate
    if minplate: query += '.filter(datadb.Cube.plate >= {0})'.format(minplate)
    if maxplate: query += '.filter(datadb.Cube.plate <= {0})'.format(maxplate)

    # MJD
    if minmjd: query += ".join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MJDMIN',\
        datadb.FitsHeaderValue.value >= {0})".format(minmjd)
    if maxmjd: query += ".join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MJDMAX',\
        datadb.FitsHeaderValue.value <= {0})".format(maxmjd) 

    # Plate Type
    if type != 'any':
        if type == 'galaxy':
            query += ".join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MNGTRG1',\
                datadb.FitsHeaderValue.value != '0')"
        elif type =='anc':
            query += ".join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MNGTRG3',\
                datadb.FitsHeaderValue.value != '0')"
        elif (type=='sky' or type=='stellar'):
            query += ".join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MNGTRG2',\
                datadb.FitsHeaderValue.value != '0')" 
                
    # IFU
    if ifu != 'Any':
        if ifu == '7':
            query += ".join(datadb.IFUDesign).filter(datadb.IFUDesign.name.like('%{0}%'),~datadb.IFUDesign.name.like('%127%'),~datadb.IFUDesign.name.like('%37%'))".format(ifu)
        else:
            query += ".join(datadb.IFUDesign).filter(datadb.IFUDesign.name.like('%{0}%'))".format(ifu)

    # Version
    if tag != 'Any':
        query += ".join(datadb.PipelineInfo,datadb.PipelineVersion).filter(datadb.PipelineVersion.version=={0})".format(tag)
    
    #query = '\n.'.join(query.split('.'))                  
    return query
    
def buildQuery(session=None, minplate=None, maxplate=None, minmjd=None, maxmjd=None, 
                tag=gu.getMangaVersion(simple=True),type='any',ifu='any', sql=None,
                user=None, keyword=None, date=None, cat='any', issues='any'):
    ''' Build the SQLalchemy query'''
    
    query = session.query(datadb.Cube)
    
    # Plate
    if minplate: query = query.filter(datadb.Cube.plate >= minplate)
    if maxplate: query = query.filter(datadb.Cube.plate <= maxplate)
    
    # MJD
    if minmjd: query = query.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MJDMIN',
        datadb.FitsHeaderValue.value >= str(minmjd))
    if maxmjd: query = query.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MJDMAX',
        datadb.FitsHeaderValue.value <= str(maxmjd))    
    
    # Plate Type
    if type != 'any':
        if type == 'galaxy':
            query = query.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MNGTRG1',
                datadb.FitsHeaderValue.value != '0')
        elif type =='anc':
            query = query.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MNGTRG3',
                datadb.FitsHeaderValue.value != '0')
        elif (type=='sky' or type=='stellar'):
            query = query.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MNGTRG2',
                datadb.FitsHeaderValue.value != '0')                  
    
    # IFU
    if ifu != 'Any':
        if ifu == '7':
            query = query.join(datadb.IFUDesign).filter(datadb.IFUDesign.name.like('%{0}%'.format(ifu)),~datadb.IFUDesign.name.like('%127%'),~datadb.IFUDesign.name.like('%37%'))
        else:
            query = query.join(datadb.IFUDesign).filter(datadb.IFUDesign.name.like('%{0}%'.format(ifu)))
        
    # Version
    if tag != 'Any':
        query = query.join(datadb.PipelineInfo,datadb.PipelineVersion).filter(datadb.PipelineVersion.version==tag)
        
    return query

def getFormParams():
    ''' Get the form parameters '''
    
    minplate = valueFromRequest(key='minplate',request=request, default=None)
    maxplate = valueFromRequest(key='maxplate',request=request, default=None)
    minmjd = valueFromRequest(key='minmjd',request=request, default=None)
    maxmjd = valueFromRequest(key='maxmjd',request=request, default=None)
    type = valueFromRequest(key='type',request=request, default='any')
    ifu = valueFromRequest(key='ifu',request=request, default='any')
    tag = valueFromRequest(key='tag',request=request, default=gu.getMangaVersion(simple=True))
    sql = valueFromRequest(key='sqlinput',request=request, default=None)
    user = valueFromRequest(key='user',request=request, default=None)
    keyword = valueFromRequest(key='keyword',request=request, default=None)
    date = valueFromRequest(key='date',request=request, default=None)
    cat = valueFromRequest(key='cat',request=request, default='any')
    issues = valueFromRequest(key='issues',request=request, default='any')
    
    form={}
    form['minplate'] = minplate
    form['maxplate'] = maxplate
    form['minmjd'] = minmjd
    form['maxmjd'] = maxmjd
    form['type'] = type
    form['tag'] = tag
    form['ifu'] = ifu
    form['sql'] = sql
    form['user'] = user
    form['keyword'] = keyword
    form['date'] = date 
    form['cat'] = cat
    form['issues'] = issues.split(',')
    
    return form
    
search_page = flask.Blueprint("search_page", __name__)

@search_page.route('/manga/writeFits',methods=['GET','POST'])
@search_page.route('/writeFits',methods=['GET','POST'])
def writeFits():
    ''' Write a FITs file from the data table '''
    
    name = valueFromRequest(key='fitsname',request=request, default=None)
    data = valueFromRequest(key='hiddendata',request=request, default=None)
    if name: name = name+'.fits' if '.fits' not in name else name
    if not name: name = 'myDataTable.fits'
    
    # Build astropy table and convert to FITS
    table = processTableData(data)
    fitstable = fits.BinTableHDU.from_columns(np.array(table))
    fitstable.add_checksum()
    prihdr = fits.Header()
    fitshead = fits.PrimaryHDU(header=prihdr)
    newhdu = fits.HDUList([fitshead,fitstable])
    size = fitshead.filebytes() + fitstable.filebytes()
    
    # write to temporary file and read back in as binary
    tmpfile = tempfile.NamedTemporaryFile()
    newhdu.writeto(tmpfile.name)
    data = tmpfile.read()
    tmpfile.close()
    
    # create and return the response
    response = Response()
    response.data =  data
    response.headers["Content-Type"] = "application/fits"
    response.headers["Content-Description"] = "File Transfer"
    response.headers["Content-Disposition"] = "attachment; filename={0}".format(name)
    response.headers["Content-Transfer-Encoding"] = "binary"
    response.headers["Expires"] = "0"
    response.headers["Cache-Control"] = "must-revalidate, post-check=0, pre-check=0"
    response.headers["Pragma"] = "public"
    response.headers["Content-Length"] = "{0}".format(size);
    
    output = 'Succesfully wrote {0}'.format(name)
    
    return response

@search_page.route('/manga/getsql',methods=['GET'])
@search_page.route('/getsql',methods=['GET'])
def getSQL():
    ''' Get the sql with the current form data '''
    
    session=db.Session
    form = getFormParams()
    query = buildQuery(session,**form)
    sql = buildSQLString(**form)
    rawsql = str(query.statement.compile(dialect=postgresql.dialect(),compile_kwargs={'literal_binds':True}))    
    rawsql = rawsql.splitlines()    
    
    return jsonify(result={'text':'sql results','rawsql':rawsql,'sql':sql})

@search_page.route('/search.html', methods=['GET','POST'])
def search():
    ''' Documentation here. '''
    
    session = db.Session() 
    search = {}
    
    # set default cubes to None
    dosearch = valueFromRequest(key='search_form',request=request, default=None)
    dosql = valueFromRequest(key='directsql_form',request=request, default=None)
    docomm = valueFromRequest(key='comment_form',request=request, default=None)
    if not dosearch: search['cubes'] = None
    search['dosearch'] = dosearch
    search['dosql'] = dosql
    search['docomm'] = docomm
    search['form'] = None
    
    # lists
    search['ifulist'] = [127, 91, 61, 37, 19, 7]
    search['types'] = ['any','galaxy','stellar','sky','anc']
    search['feedback'] = Feedback(current_session)

    # pipeline versions
    vers = session.query(datadb.PipelineVersion).all()
    search['versions'] = sorted([v.version for v in vers], reverse=True)
    search['current'] = search['versions'][0]
    
    # get form params
    if dosearch or dosql or docomm:
        form = getFormParams()
        search['form'] = form
        print(form)
    
    # do sql query
    if dosql:
        result = doRawSQLSearch(form['sql'])
        search['goodsql'] = True if result else False
        search['sqlresult'] = result
        cubes = [row for row in result] if result and result.rowcount != 0 else None
        if cubes:
            search['cubes'] = cubes
            # build table
            cubetable,displayCols = buildSQLTable(cubes)
            search['cubetable'] = cubetable
            search['cols'] = displayCols
            search['keys'] = cubetable.keys()
                    
    # do cube search query
    if dosearch:
        # build and submit query
        query = buildQuery(session, **form)
        cubes = query.order_by(datadb.Cube.pk).all()

        if cubes: 
            search['cubes'] = cubes 
            # build table needs
            cubetable,displayCols = buildTable(cubes)
            search['cubetable'] = cubetable
            search['cols'] = displayCols
            search['keys'] = cubetable.keys()
            search['flags'] = makeQualNames(cubetable['drp3qual'],stage='3d')
    
    # do comment search query
    if docomm:
        #cubes = sqlalchemy comment rows
        #cubetable - astropy table of comment results
        #cols = initial display columns in table
        #keys = list of all columns in table
        pass
    
    return render_template("search.html", **search)

