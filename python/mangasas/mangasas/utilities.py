
''' General Utilities for MaNGA SAS'''

import json
from ast import literal_eval
from manga_utils import generalUtils as gu
from astropy.table import Table
from model.database import db
import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

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
        'rfwhm':float,'ifwhm':float,'zfwhm':float, 'ebvgal':float, 'datered':long,'cartid':long,
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
    
    
    