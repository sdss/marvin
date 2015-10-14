#!/usr/bin/python

import os, glob, random

import flask
from flask import request, render_template, send_from_directory, current_app, session as current_session
from manga_utils import generalUtils as gu
from collections import OrderedDict
from ..utilities import setGlobalSession

from ..model.database import db

import sdss.internal.database.utah.mangadb.DataModelClasses as datadb

image_page = flask.Blueprint("image_page", __name__)

@image_page.route('/images.html', methods=['GET'])
def images():
    ''' Images page '''
    
    session = db.Session()    
    images = {}

    setGlobalSession()
    images['version'] = current_session['currentver']
    
    # find and grab all images ; convert paths to sas paths
    redux = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),version)
    sasredux = os.path.join(os.getenv('SAS_REDUX'),version)
    good = os.path.isdir(redux)
    images['good'] = good
    imagedir = os.path.join(redux,'*','stack','images')
    imagelist = glob.glob(os.path.join(imagedir,'*.png'))
    imagelist = [os.path.join(os.getenv('SAS_URL'),sasredux,i.rsplit('/',4)[1],'stack/images',i.split('/')[-1]) for i in imagelist]
    
    
    # make unique image dictionary
    imdict = OrderedDict()
    for image in imagelist:
        plate = image.rsplit('/',4)[1]
        name = image.rsplit('/',4)[-1].split('.')[0]
        plateifu = '{0}-{1}'.format(plate,name)
        imdict[plateifu] = image
    images['imdict'] = imdict
    


    return render_template("images.html", **images)
