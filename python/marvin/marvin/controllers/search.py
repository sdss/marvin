#!/usr/bin/python

import os, re

import flask, sqlalchemy, json, tempfile
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import text
from sqlalchemy import or_
from sqlalchemy.sql.expression import func
from flask import request, render_template, send_from_directory, current_app, jsonify, Response
from flask import session as current_session
from operator import le,ge,gt,lt,eq,ne
from manga_utils import generalUtils as gu
from astropy.table import Table
from astropy.io import fits
from collections import defaultdict
import numpy as np

from ..model.database import db
from ..utilities import makeQualNames, processTableData,getMaskBitLabel, \
setGlobalVersion, getDRPVersion, getDAPVersion

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

try: from inspection.marvin import Inspection
except: from marvin.inspection import Inspection

try:
    from . import valueFromRequest, processRequest
except ValueError:
    pass 


def selectByMangaTarget(query,name,type):

    # look to see if the tables are already joined
    try:
        fitsheadin = 'fits_header' in str(query._from_obj[0])
    except: 
        fitsheadin = False

    # if not, make the join
    if not fitsheadin: query = query.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword)

    # filter on criteria
    if type == 'in':
        query = query.filter(datadb.FitsHeaderKeyword.label==name,datadb.FitsHeaderValue.value != '0')
    elif type == 'out':
        query = query.filter(datadb.FitsHeaderKeyword.label==name,datadb.FitsHeaderValue.value == '0')
    return query

def makeNSAList():
    ''' Make the initial NSA list '''

    nsanames = ['redshift','absMag','absMag_el','amivar_el','stellar mass','stellar mass_el','ba','phi','extinction','petro th50',
    	'petro th50_el','petroflux', 'petroivar', 'petroflux_el','petroivar_el','sersic b/a', 'sersic n', 'sersic phi', 'sersic th50', 
    	'sersicflux', 'sersicivar']
    nsalist = [{'id':i+1,'name':name} for i,name in enumerate(nsanames)]
    nsamagids = [2,3,4,9,12,13,14,15,20,21]

    return nsalist, nsamagids
        
def parseNSA(nsatext):
    ''' Parse the NSA search form list '''
    
    nsalist = flask.g.get('nsalist',None)
    nsamagids = flask.g.get('nsamagids',None)
    bands = ['f','n','u','g','r','i','z']
    magids = nsamagids
        
    coldict = {'redshift':'redshift','absMag':'absmag','absMag_el':'absmag_el','amivar_el':'amivar_el',
        'stellar mass':'mstar','stellar mass_el':'mstar_el','ba':'ba','phi':'phi','extinction':'extinction',
    	'petro th50_el':'petro_th50_el','petro th50':'petro_th50','petroflux':'petroflux', 
    	'petroivar':'petroflux_ivar', 'petroflux_el':'petroflux_el', 
    	'petroivar_el':'petroflux_ivar_el','sersic b/a':'sersic_ba', 'sersic n':'sersic_n', 
    	'sersic phi':'sersic_phi', 'sersic th50':'sersic_th50', 
    	'sersicflux':'sersicflux', 'sersicivar':'sersicflux_ivar'}
    
    # build full list with sample columns
    fulllist = [] 
    count=1
    for row in nsalist:
        if row['id'] in magids:
            for j,mag in enumerate(bands):
                fulllist.append({'id':count+j,'oid':row['id'],'name':row['name'],'samplecol':'nsa_{0}_{1}'.format(coldict[row['name']],mag)})
            count+=6
        else:
            fulllist.append({'id':count,'oid':row['id'],'name':row['name'],'samplecol':'nsa_{0}'.format(coldict[row['name']])})    
        count+=1
        
    # filter out the parameters not selected
    finalist = {fulllist[i]['samplecol']:nsatext[i] for i,item in enumerate(nsatext) if item}
    flask.g.nsamap = [{'id':row['id'],'oid':row['oid']} for row in fulllist]
    
    return finalist
    
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
        for key,val in d.iteritems():
            cubedict[key].append(val)
    cubedict = dict(cubedict.iteritems())
    
    # convert to Table
    cubetable = Table(cubedict)[cols]
    
    return cubetable, displayCols

def buildTable(cubes):
    ''' Build the data table as a dictionary '''
    
    # build table column list

    # from header
    cols = ['plate','ifudesign','mangaid','versdrp2','versdrp3','verscore','versutil','platetyp','srvymode',
    'objra','objdec','ifuglon','ifuglat','ebvgal','nexp','exptime','drp3qual','bluesn2','redsn2','harname','frlplug',
    'cartid','designid','cenra','cendec','airmsmin','airmsmed','airmsmax','seemin','seemed','seemax','transmin',
    'transmed','transmax','mjdmin','mjdmed','mjdmax','gfwhm','rfwhm','ifwhm','zfwhm','mngtarg1','mngtarg2',
    'mngtarg3','catidnum','plttarg']
    
    # sample db cols
    sampcols = [s for s in datadb.Sample().cols if 'nsa' in s]

    # combine
    cols.extend(sampcols)
    displayCols=['plate','ifudesign','harname','mangaid','drp3qual','versdrp3','verscore','mjdmax','objra', 'objdec','bluesn2','redsn2','nexp','exptime']
    
    cubedict=defaultdict(list)
    for cube in cubes:
        try: hdr = json.loads(cube.hdr[0].header)
        except: hdr = cube.header_to_dict()
        
        for col in cols:
            if col=='plate': cubedict['plate'].append(cube.plate)
            elif col=='mangaid': cubedict['mangaid'].append(cube.mangaid)
            elif col=='designid': cubedict['designid'].append(cube.designid)
            elif col=='ifudesign': cubedict['ifudesign'].append(cube.ifu.name)
            elif 'mngtarg' in col: 
                try: cubedict[col].append(hdr[''.join(col.split('a')).upper()])
                except: cubedict[col].append(None) 
            else: 
                if col.upper() in hdr:
                    # grab from header
                    try: cubedict[col].append(hdr[col.upper()])
                    except: cubedict[col].append(None)
                elif col in sampcols:
                    # grab from sample table
                    try: cubedict[col].append(cube.sample[0].__getattribute__(col))                
                    except: cubedict[col].append(None)
                else:
                    cubedict[col].append(None)
                
    cubetable = Table(cubedict)
    cubetable = cubetable[cols]
    
    return cubetable, displayCols
    

def buildSQLString(minplate=None, maxplate=None, minmjd=None, maxmjd=None, mangaid=None,
                tag=gu.getMangaVersion(simple=True),type='any',ifu='any', sql=None,
                user=None, keyword=None, date=None, cat='any', issues='any', ra=None, 
                dec=None, searchrad=None, radecmode=None, nsatext=None,search_form=None,tagids=None,defaultids=None):
    ''' Build a string version of the SQLalchemy query'''
    
    query = 'session.query(datadb.Cube)'

    # Plate
    if minplate: query += '.filter(datadb.Cube.plate >= {0})'.format(minplate)
    if maxplate: query += '.filter(datadb.Cube.plate <= {0})'.format(maxplate)

    # MJD
    if minmjd and not maxmjd: query += ".join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MJDMIN',\
        datadb.FitsHeaderValue.value >= {0})".format(minmjd)
    if maxmjd and not minmjd: query += ".join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MJDMAX',\
        datadb.FitsHeaderValue.value <= {0})".format(maxmjd) 
    if minmjd and maxmjd: query += ".join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(or_(datadb.FitsHeaderKeyword.label=='MJDMIN',\
    	datadb.FitsHeaderKeyword.label=='MJDMAX'), datadb.FitsHeaderValue.value >= {0},datadb.FitsHeaderValue.value <= {1})".format(minmjd,maxmjd)        
    
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

    # RA, Dec Search
    if ra or dec:
        # Between search mode
        if radecmode == 'between':
            if ra:
                try: ralow,raup = ra.split(',')
                except ValueError: ralow,raup = ra.split(' ') 
                query += ".filter(datadb.Cube.ra >= {0}, datadb.Cube.ra <= {1})".format(ralow,raup)
            if dec:
                try: declow,decup = dec.split(',')
                except ValueError: declow,decup = dec.split(' ')
                query += ".filter(datadb.Cube.dec >= {0}, datadb.Cube.dec <= {1})".format(declow,decup)
        # Cone search mode
        if radecmode == 'cone':
            radius = 1.0 if not searchrad else float(searchrad)
            ra = float(ra) if ra.isdigit() else None
            dec = float(dec) if dec.isdigit() else None
            query += ".filter(func.q3c_radial_query(datadb.Cube.ra,datadb.Cube.dec,{0},{1},{2}))".format(ra,dec,radius)                       
                
    # NSA
    if any(nsatext):
	    nsapars = parseNSA(nsatext)
	    cols = datadb.Sample().cols
	    opdict = {'<=':le,'>=':ge,'>':gt,'<':lt,'!=':ne,'=':eq}
	    query += ".join(datadb.Sample)"
	    # loop over params and build query
	    for key,value in nsapars.iteritems():
	        iscompare = any([s in value for s in opdict.keys()])

	        # do operator query or do range query
	        if iscompare:
	            # separate operator and value
	            value.strip()
	            try: ops,number = value.split()
	            except ValueError: 
	                match = re.match(r"([<>=!]+)([0-9.]+)", value, re.I)
	                if match:
	                    ops = match.groups()[0]
	                    number = match.groups()[1]    
	            op = opdict[ops]
	            	            
	            # build query
	            query += ".filter(datadb.Sample.{1} {2} {0}))".format(number,key,ops)
	        else:
	            # try splits on dash, comma, or space
	            try: low,up = value.split('-')
	            except ValueError:
	               try: low,up = value.split(',')
	               except ValueError:
	                   try: low,up = value.split()
	                   except ValueError: low,upp = [None,None]
	            # build query
	            if low:       
	                query += ".filter(datadb.Sample.{2} >= {0}, datadb.Sample.{2} <= {1})".format(low,up,key)

    # MaNGA-ID
    if mangaid:
        query += ".filter(datadb.Cube.mangaid = {0})".format(mangaid)  

    return query
    
def buildQuery(session=None, minplate=None, maxplate=None, minmjd=None, maxmjd=None, mangaid=None,
                tag=gu.getMangaVersion(simple=True),type='any',ifu='any', sql=None,
                user=None, keyword=None, date=None, cat='any', issues='any', ra=None, 
                dec=None, searchrad=None, radecmode=None, nsatext=None, search_form=None, tagids=None, defaultids=None):
    ''' Build the SQLalchemy query'''
    
    query = session.query(datadb.Cube)
    
    # Plate
    if minplate: query = query.filter(datadb.Cube.plate >= minplate)
    if maxplate: query = query.filter(datadb.Cube.plate <= maxplate)
    
    # MJD
    if minmjd and not maxmjd: query = query.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MJDMIN',
        datadb.FitsHeaderValue.value >= str(minmjd))
    if maxmjd and not minmjd: query = query.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(datadb.FitsHeaderKeyword.label=='MJDMAX',
        datadb.FitsHeaderValue.value <= str(maxmjd))
    if minmjd and maxmjd: query = query.join(datadb.FitsHeaderValue,datadb.FitsHeaderKeyword).filter(or_(datadb.FitsHeaderKeyword.label=='MJDMIN',
    	datadb.FitsHeaderKeyword.label=='MJDMAX'), datadb.FitsHeaderValue.value >= str(minmjd),datadb.FitsHeaderValue.value <= str(maxmjd))    
    
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
    
    # RA, Dec Search
    if ra or dec:
        # Between search mode
        if radecmode == 'between':
            if ra:
                try: ralow,raup = ra.split(',')
                except ValueError: ralow,raup = ra.split(' ') 
                query = query.filter(datadb.Cube.ra >= ralow, datadb.Cube.ra <= raup)
            if dec:
                try: declow,decup = dec.split(',')
                except ValueError: declow,decup = dec.split(' ')
                query = query.filter(datadb.Cube.dec >= declow, datadb.Cube.dec <= decup)
        # Cone search mode
        if radecmode == 'cone':
            radius = 1.0 if not searchrad else float(searchrad)
            ra = float(ra) if ra.isdigit() else None
            dec = float(dec) if dec.isdigit() else None
            query = query.filter(func.q3c_radial_query(datadb.Cube.ra,datadb.Cube.dec,ra,dec,radius))             
    
    # NSA
    if any(nsatext):
	    nsapars = parseNSA(nsatext)
	    cols = datadb.Sample().cols
	    opdict = {'<=':le,'>=':ge,'>':gt,'<':lt,'!=':ne,'=':eq}
	    query = query.join(datadb.Sample)
	    # loop over params and build query
	    for key,value in nsapars.iteritems():
	        iscompare = any([s in value for s in opdict.keys()])

	        # do operator query or do range query
	        if iscompare:
	            # separate operator and value
	            value.strip()
	            try: ops,number = value.split()
	            except ValueError: 
	                match = re.match(r"([<>=!]+)([0-9.]+)", value, re.I)
	                if match:
	                    ops = match.groups()[0]
	                    number = match.groups()[1]    
	            op = opdict[ops]
	            	            
	            # build query
	            query = query.filter(op(datadb.Sample.__table__.columns.__getitem__(cols[cols.index(key)]),number))
	        else:
	            # try splits on dash, comma, or space
	            try: low,up = value.split('-')
	            except ValueError:
	               try: low,up = value.split(',')
	               except ValueError:
	                   try: low,up = value.split()
	                   except ValueError: low,up = [None,None]
	            # build query
	            if low:       
	                query = query.filter(datadb.Sample.__table__.columns.__getitem__(cols[cols.index(key)]) >= low, datadb.Sample.__table__.columns.__getitem__(cols[cols.index(key)]) <= up)
	# MaNGA-ID
    if mangaid:
        query = query.filter(datadb.Cube.mangaid == mangaid)

    # Set Defaults
    #print('defaults',defaultids)
    #if defaultids != 'any':
    #    ids = [int(d.split('def')[1]) for d in defaultids.split(',')]
    #    num = len(ids)
    #    # {1:'Primary',2:'Primary,color-enhanced',3:'Secondary',4:'Ancillary',5:'Stellar Library',6:'Flux Standard Stars'}
    #    defaults = flask.g.get('defaults',None)
    #    query = selectByMangaTarget(query,'MNGTRG1','in') if 1 and 2 and 3 in ids else selectByMangaTarget(query,'MNGTRG1','out')
    #    query = selectByMangaTarget(query,'MNGTRG3','in') if 4 in ids else selectByMangaTarget(query,'MNGTRG3','out')
    #    query = selectByMangaTarget(query,'MNGTRG2','in') if 5 in ids else selectByMangaTarget(query,'MNGTRG2','out')
    #    if 6 in ids: query = query.join(datadb.IFUDesign,datadb.Fibers,datadb.TargetType).filter(datadb.TargetType.label == 'standard')
    #    else: query = query.join(datadb.IFUDesign,datadb.Fibers,datadb.TargetType).filter(datadb.TargetType.label != 'standard')
            
    return query

def getFormParams():
    ''' Get the form parameters '''

    form = processRequest(request=request)
    if 'tagids' in form:
        print('form tags', form['tagids'])
    if 'issues' in form:
        print('form issues', form['issues'])
    if 'dapissues' in form:
        print('form dapissues', form['dapissues'])

    if 'defaultids' in form:
        if form['defaultids'] != 'any':
            ids = [int(d.split('def')[1]) for d in form['defaultids'].split(',')]
            if ids and current_session['defaults'] != ids:
                current_session['defaults'] = ids
        else:
            current_session['defaults'] = ['any']

    print('search form',form)
            
    return form
    
search_page = flask.Blueprint("search_page", __name__)

@search_page.route('/marvin/writeFits',methods=['GET','POST'])
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

@search_page.route('/marvin/getsql',methods=['GET','POST'])
@search_page.route('/getsql',methods=['GET','POST'])
def getSQL():
    ''' Get the sql with the current form data '''
    
    session=db.Session
    nsalist = makeNSAList()
    flask.g.nsalist = nsalist
    form = getFormParams()
    query = buildQuery(session,**form)
    sql = buildSQLString(**form)
    rawsql = str(query.statement.compile(dialect=postgresql.dialect(),compile_kwargs={'literal_binds':True}))    
    rawsql = rawsql.splitlines()    
    
    return jsonify(result={'text':'sql results','rawsql':rawsql,'sql':sql})

@search_page.route('/search/', methods=['GET','POST'])
@search_page.route('/marvin/search/', methods=['GET','POST'])
def search():
    ''' Documentation here. '''
    
    session = db.Session() 
    search = {}
    search['title'] = "Marvin | Search"
    
    # set global version    
    try: version = current_session['currentver']
    except: 
        setGlobalVersion()
        version = current_session['currentver']

    # set default search options
    search['defaults'] = {1:'Primary',2:'Primary,color-enhanced',3:'Secondary',4:'Ancillary',5:'Stellar Library',6:'Flux Standard Stars'}
    flask.g.defaults = search['defaults']
    current_session['defaults'] = [1,2,3,4] if 'defaults' not in current_session else current_session['defaults']

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
    search['inspection'] = inspection = Inspection(current_session)
    inspection.set_options()
    inspection.set_panels()
    inspection.retrieve_alltags(ids=True)
    nsalist,nsamagids = makeNSAList()
    flask.g.nsalist = nsalist
    flask.g.nsamagids = nsamagids
    search['nsalist'] = nsalist
    search['nsamagids'] = nsamagids
    search['nsamap'] = []

    # pipeline versions
    search['versions'] = getDRPVersion() 
    search['dapversions'] = getDAPVersion()
    search['current'] = version
    
    # get form params
    if dosearch or dosql or docomm:
        form = getFormParams()
        search['form'] = form
    
    # do sql query
    if dosql:
        result = doRawSQLSearch(form['sqlinput'])
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
        
        if 'tagids' in form:
            if form['tagids'] != 'any':
                cubes = inspection.refine_cubes_by_tagids(tagids=form['tagids'], cubes=cubes)
            
        if cubes: 
            search['cubes'] = cubes 
            # build table needs
            cubetable,displayCols = buildTable(cubes)
            search['cubetable'] = cubetable
            search['cols'] = displayCols
            search['keys'] = cubetable.keys()
            #search['flags'] = makeQualNames(cubetable['drp3qual'],stage='3d')
            search['flags'] = getMaskBitLabel(cubetable['drp3qual'])
    
    # do comment search query
    if docomm:
        if 'inspection_counter' in inspection.session: current_app.logger.info("Inspection Counter %r" % inspection.session['inspection_counter'])
        current_app.logger.warning('Inspection> search parameters %r' % form)
        inspection.set_search_parameters(form=form)
        inspection.retrieve_searchcomments()
        inspection.retrieve_dapqasearchcomments()
        result = inspection.result()
        if inspection.status: current_app.logger.warning('Inspection> GET searchcomments: {0}'.format(result))
        else: current_app.logger.warning('Inspection> GET searchcomments: No Results')

    return render_template("search.html", **search)

