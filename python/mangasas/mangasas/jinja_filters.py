#!/usr/bin/python

'''
This file contains all custom Jinja2 filters.
'''

from manga_utils import generalUtils as gu


def split(string, delim=None):
    '''Split a string based on a delimiter'''
    
    if not delim: delim=' '
    
    return string.split(delim) if string else None
    
    
def colorCode(value, column):
    ''' Return status code for class to color-code it '''
    
    state = None
    if 'status' in column and value=='fault': state='danger'
    if 'apocomp' in column:
        if value=='In': state='success'
        if value=='Out': state='danger'
    if 'complete' in column:
        state = 'success' if value=='Yes' else 'danger' if value=='No' else ''
    
    return state

def popOver(value,column,location=None,title=False,content=False):
    ''' Return a tooltip for a given column '''
    
    tip = None
    
    # DRP2/3 QUALITY FLAGS
    name = 'MANGA_{0}'.format(column.upper())
    if 'qual' in column:        
        flag = gu.getSDSSFlagName(value,name=name)        
        tip = 'popover'
    if tip and location: tip=location
    if tip and title: tip = 'Flags'
    if tip and content: tip=flag
    
    return tip
    
def setFlag(value,column):
    ''' set quality flags '''
    
    flag = None
    name = 'MANGA_{0}'.format(column.upper())
    if 'qual' in column:
        flag = gu.getSDSSFlagName(value,name=name)
        
    return flag
    
def getMPL(value):
    ''' Define the MPL version '''
    
    name = value
    
    if value == 'v1_0_0': name='{0} (MPL 1)'.format(value)
    if value == 'v1_1_2': name='{0} (MPL 2)'.format(value)
    
    return name

def filterForm(value,name,form):
    ''' Filter the form display values based on form parameters '''

    if form:
        if form[name]: value = form[name]

    return value