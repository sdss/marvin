
''' General Utilities for MaNGA SAS'''

import json, os, glob
from flask import session as current_session
from ast import literal_eval
from manga_utils import generalUtils as gu
from astropy.table import Table
from model.database import db
from jinja_filters import getMPL
import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

def getMaskBitLabel(bits):
    ''' Return a list of flag names '''
    
    session = db.Session()
    flagnames=[]
    for bit in bits:
        if bit: 
            try: 
                masklabels = session.query(datadb.MaskLabels).filter_by(maskbit=long(bit)).one()
                labels = json.loads(masklabels.labels)
                labels = str(','.join(labels))
            except:
                labels = None
        else: labels = None
        flagnames.append(labels)

    return flagnames

def makeQualNames(bits, stage='2d'):
    ''' Return list containing the quality flag names '''
     
    name = 'MANGA_DRP2QUAL' if stage=='2d' else 'MANGA_DRP3QUAL'
    #flagnames = [gu.getSDSSFlagName(bit,name=name) for bit in bits]
    flagnames = [getFlags(bit,name) for bit in bits]
    return flagnames
    
def getTargNames(bits, type=1):
    ''' Return list containing MaNGA Target Flags '''
    
    name = 'MANGA_TARGET1' if type==1 else 'MANGA_TARGET2' if type==2 else 'MANGA_TARGET3'
    #flagnames = [gu.getSDSSFlagName(bit,name=name) for bit in bits]
    flagnames = [getFlags(bit,name) for bit in bits]
    return flagnames

def getFlags(bits,name):
    ''' Get the labels for a given bitmask, from database '''

    session = db.Session()	    
    
    # if bits not a digit, return None
    if not str(bits).isdigit(): return 'NULL'
    else: bits=int(bits)
    
    # Convert the integer value to list of bits
    bitlist=[int(i) for i in '{0:08b}'.format(bits)]
    bitlist.reverse()
    indices = [i for i,bit in enumerate(bitlist) if bit]	    
    
    labels=[]
    for i in indices:
        maskbit = session.query(datadb.MaskBit).filter_by(flag=name,bit=i).one()
        labels.append(str(maskbit.label))    
    
    if not labels: labels=None
    
    return labels
    
    
def getColumnTypes(columns):
    ''' Return the data types for table columns '''
    
    typedict = {'nexp':int, 'exptime':float, 'bluesn2':float, 'redsn2':float,
        'airmsmin':float, 'airmsmed':float, 'airmsmax':float, 'mjdmin':long, 
        'mjdmed':long, 'mjdmax':long, 'objra':float, 'objdec':float,
        'ifudesign':str, 'harname':str, 'cenra':float, 'cendec':float,
        'frlplug':int, 'mangaid':str, 'ufwhm':float,'gfwhm':float,
        'rfwhm':float,'ifwhm':float,'zfwhm':float, 'ebvgal':float, 'datered':long,'cartid':str,
        'versdrp3':str,'verscore':str, 'versutil':str, 'drp3qual':long,
        'objglon':float, 'objglat':float, 'designid':long,'catidnum':long,
        'seemin':float, 'seemed':float, 'seemax':float,'transmin':float, 'transmed':float, 
        'transmax':float, 'mngtarg1':long, 'mngtarg2':long, 'plttarg':str,'mjdred':str,
        'mngtarg3':long, 'mangaid':str, 'plate':long, 'platetyp':str, 'srvymode':str, 'versdrp2':str}
    
    types = [typedict[col] if col in typedict.keys() else str for col in columns]
        
    return types, typedict    
    
def processTableData(tableobj):
    ''' Process a JSON object table and convert to an Astropy Table '''
    
    table = (json.loads(tableobj))
    nrows = len(table)

    keys = [k for k in table[0].keys() if '_class' not in k]
    types,typedict = getColumnTypes(keys)
   
    for row in table:
        for key,val in row.items():
            # pull the plate id from the link
            index = val.find('">')
            if index != -1: row[key] = val[index+2:index+6]
            
            # convert QUAL keys back to bits
            if 'qual' in key: row[key] = gu.getSDSSFlagBit(literal_eval(row[key]),name='MANGA_DRP3QUAL') if row[key] != 'NULL' else None
            
            # reset NULL values to the right type
            if val == None or val=='None' or val=='NULL':
                row[key] = gu.getNullVal(typedict[key])
        
        # Delete the _class key
        tmp = row.pop('_class',None) 
    
    newtable = Table(table,dtype=types,names=keys)    
    
    return newtable
    
    
def getDRPVersion():
    ''' Get DRP version to use during MaNGA SAS '''

	# DRP versions
    session = db.Session()    
    vers = session.query(datadb.PipelineVersion).filter(datadb.PipelineVersion.version.like('%v%')).order_by(datadb.PipelineVersion.version.desc()).all()
    versions = [v.version for v in vers]
    
    return versions    

def getDAPVersion():
    ''' Get DAP version to use during MaNGA SAS '''
    
    # DAP versions
    session = db.Session()    
    vers = session.query(datadb.PipelineVersion).join(datadb.PipelineInfo,datadb.PipelineName).\
    filter(datadb.PipelineName.label=='DAP',datadb.PipelineVersion.version.like('%v%')).\
    order_by(datadb.PipelineVersion.version.desc()).all()
    versions = [v.version for v in vers]
    
    return versions+['NA']

def setMPLVersion(mplver):
    ''' set the versions based on MPL '''
    
    mpl = getMPL(mplver)
    drpver,dapver = mpl.split(':')[1].strip().split(', ')
    current_session['currentver'] = drpver
    current_session['currentdapver'] = dapver if dapver != 'NA' else None
    
def setGlobalVersion():
    ''' set the global version '''

    # set MPL version
    try: mplver = current_session['currentmpl']
    except: mplver = None
    if not mplver: current_session['currentmpl']='MPL-3'
    
    # set version mode
    try: vermode = current_session['vermode']
    except: vermode = None 
    if not vermode: current_session['vermode']='MPL'
    
    # initialize
    if 'MPL' in current_session['vermode']:
        setMPLVersion(current_session['currentmpl'])
    
    # set global DRP version
    try: versions = current_session['versions'] 
    except: versions = getDRPVersion()
    current_session['versions'] = versions
    try: drpver = current_session['currentver']
    except: drpver = None
    if not drpver: 
        realvers = [ver for ver in versions if os.path.isdir(os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),ver))]
        current_session['currentver'] = realvers[0]
        
    # set global DAP version
    try: dapversions = current_session['dapversions'] 
    except: dapversions = getDAPVersion()
    current_session['dapversions'] = dapversions
    try: ver = current_session['currentdapver']
    except: ver = None
    if not ver: 
        realvers = [ver for ver in versions if os.path.isdir(os.path.join(os.getenv('MANGA_SPECTRO_ANALYSIS'),current_session['currentver'],ver))]
        current_session['currentdapver'] = realvers[0]    


def getImages(plate=None,version=None,ifuname=None):
    ''' grab all PNG IFU images from a given directory '''
    
    # set version
    if not version:
        try: version = current_session['currentver']
        except: 
            setGlobalVersion()
            version = current_session['currentver']
            
    # build paths 
    if plate:
        sasredux = os.path.join(os.getenv('SAS_REDUX'),version,str(plate))
        redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version,str(plate))
    else:
        sasredux = os.path.join(os.getenv('SAS_REDUX'),version)
        redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version,'*')    
    try: sasurl = os.getenv('SAS_URL')
    except: sasurl= None    
    
    # get images and replace path with sas path
    imagedir = os.path.join(redux,'stack','images')
    images = glob.glob(os.path.join(imagedir,'*.png'))
    if plate:
        images = [os.path.join(sasurl,sasredux,'stack/images',i.split('/')[-1]) for i in images]
    else:
        images = [os.path.join(sasurl,sasredux,i.rsplit('/',4)[1],'stack/images',i.split('/')[-1]) for i in images]
    
    # filter by ifu name
    if ifuname:
        images = [im for im in images if ifuname in im]

    return images
 
def getDAPImages(plate, ifu, drpver, dapver, catkey, mode, bintype, maptype, specpaneltype,inspection, filter=True):
    ''' grab all the DAP PNG analysis plots '''
    
    # module mangadapplot (if loaded) allows for DAP QA plots outside of MANGA_SPECTRO_ANALYSIS/drpver/dapver
    # Divert away from MANGA_SPECTRO_ANALYSIS and dapver if mangadapplot loaded
    dapplotdir = os.getenv('MANGADAPPLOT_DIR') if 'MANGADAPPLOT_DIR' in os.environ else os.getenv('MANGA_SPECTRO_ANALYSIS')
    dapplotmode = os.getenv('MANGADAPPLOT_MODE') if 'MANGADAPPLOT_MODE' in os.environ else None
    dapverplotmode = '_'.join([drpver,dapplotmode]) if dapplotmode else drpver
    
    dapplotver = os.getenv('MANGADAPPLOT_VER') if 'MANGADAPPLOT_VER' in os.environ else dapver

    # build path
    print('dapplot', dapplotdir, dapverplotmode, dapplotver)
    redux = os.path.join(dapplotdir,dapverplotmode,dapplotver,str(plate),ifu,'plots')
    catdict = inspection.dapqaoptions['subfolder']

    # build sas path
    try: sasurl = os.getenv('SAS_URL')
    except: sasurl= None
    sasredux = os.getenv('SAS_ANALYSIS')
    saspath = os.path.join(sasurl,sasredux)
    
    # modify if maptype not equal to binnum
    if maptype != 'binnum':
        sasredux = os.path.join(sasredux,catdict[catkey])
        redux = os.path.join(redux,catdict[catkey])
    
    print('redux', redux)
    print('saspath',saspath)

    # grab images
    if os.path.isdir(redux):
        # build filename
        bindict = inspection.dapqaoptions['bindict']
        binname = 'BIN-{0}-00{1}'.format(bindict[bintype[:-1]],bintype[-1])
        name = 'manga-{0}-{1}-LOG{2}_{3}_*.png'.format(plate,ifu,mode.upper(),binname)
        # search for images & filter
        imgpath = os.path.join(redux,name)
        images = glob.glob(imgpath)
        if filter: images = filterDAPimages(images,maptype,catkey,bintype[:-1],specpaneltype,inspection)
        images = [os.path.join(saspath,i.split('analysis/',1)[1]) for i in images]
        msg = 'No Plots Found!' if not images else 'Success!'
    else:
        images = None
        msg = 'Not a valid DAP directory. Check version.'  

    return images, msg

def filterDAPimages(images, mapid, key,bintype,specpaneltype,inspection):
    ''' filter the DAP PNG images based on mapid and category key'''  
    
    # get the map dictionary in order to filter the image list
    mapdict = inspection.get_dapmapdict(key,bin=bintype)
    
    # Filter
    if key == 'spectra':
        name = 'spec-{0:04d}'.format(int(mapid.split('c')[1]))
        if 'single' in specpaneltype:
            images = [i for i in images if name+'.' in i]
            images.sort()
        else:
            images = [i for i in images if name+'_' in i and 'emlines' not in i]
            sortbypanel = inspection.get_panelnames('specmap',bin=bintype)
            if not any([i[0] for i in sortbypanel]): sortbypanel = None

            if sortbypanel and images:
                sortbypanel.sort(key=lambda t:t[1])
                images = list(zip(*sorted(zip(sortbypanel,images),key=lambda t:t[0][0]))[1])
    else: 
        # filter images
        if type(mapdict[mapid]) == list:
            images = [i for i in images for val in mapdict[mapid] if val in i]
        else:
            images = [img for img in images if mapdict[mapid] in img]

        # sort images
        images.sort()
        sortbypanel = inspection.get_panelnames(mapid,bin=bintype)
        if not any([i[0] for i in sortbypanel]): sortbypanel = None
        
        if sortbypanel and images:
            sortbypanel.sort(key=lambda t:t[1])
            images = list(zip(*sorted(zip(sortbypanel,images),key=lambda t:t[0][0]))[1])
    
    return images


    