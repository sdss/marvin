#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-05-02 16:16:27
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-05-02 16:16:27 by Brian Cherinka
    Last Modified On: 2016-05-02 16:16:27 by Brian

'''
from __future__ import print_function
from __future__ import division
from flask import session as current_session, request, current_app
from marvin import config
from collections import defaultdict
import flask_featureflags as feature
import re


def configFeatures(app):
    ''' Configure Flask Feature Flags '''

    app.config['FEATURE_FLAGS']['public'] = True if config.access == 'public' else False


def check_access():
    ''' Check the access mode in the session '''

    # check if on public server
    public_server = request.environ.get('PUBLIC_SERVER', None) == 'True'
    public_flag = public_server or current_app.config['FEATURE_FLAGS']['public']
    current_app.config['FEATURE_FLAGS']['public'] = public_server
    public_access = config.access == 'public'

    # ensure always in public mode when using public server
    if public_flag:
        config.access = 'public'
        return

    # check for logged in status
    logged_in = current_session.get('loginready', None)

    if not logged_in and not public_access:
        config.access = 'public'
    elif logged_in is True and public_access:
        config.access = 'collab'


def update_allowed():
    ''' Update the allowed versions '''
    mpls = list(config._allowed_releases.keys())
    versions = [{'name': mpl, 'subtext': str(config.lookUpVersions(release=mpl))} for mpl in mpls]
    return versions


def set_session_versions(version):
    ''' Set the versions in the session '''
    current_session['release'] = version
    drpver, dapver = config.lookUpVersions(release=version)
    current_session['drpver'] = drpver
    current_session['dapver'] = dapver


def updateGlobalSession():
    ''' updates the Marvin config with the global Flask session '''

    # check if mpl versions in session
    if 'versions' not in current_session:
        setGlobalSession()
    elif 'drpver' not in current_session or \
         'dapver' not in current_session:
        set_session_versions(config.release)
    elif 'release' not in current_session:
        current_session['release'] = config.release
    elif 'release' in current_session:
        if current_session['release'] not in config._allowed_releases:
            current_session['release'] = config.release

    # reset the session versions if on public site
    if feature.is_active('public'):
        current_session['versions'] = update_allowed()


def setGlobalSession():
    ''' Sets the global session for Flask '''

    current_session['versions'] = update_allowed()

    if 'release' not in current_session:
        set_session_versions(config.release)


def parseSession():
    ''' parse the current session for MPL versions '''
    drpver = current_session['drpver']
    dapver = current_session['dapver']
    release = current_session['release']
    return drpver, dapver, release


def buildImageDict(imagelist, test=None, num=16):
    ''' Builds a list of dictionaries from a sdss_access return list of images '''

    # get thumbnails and plateifus
    if imagelist:
        thumbs = [imagelist.pop(imagelist.index(t)) if 'thumb' in t else t for t in imagelist]
        plateifu = ['-'.join(re.findall('\d{3,5}', im)) for im in imagelist]

    # build list of dictionaries
    images = []
    if imagelist:
        for i, image in enumerate(imagelist):
            imdict = defaultdict(str)
            imdict['name'] = plateifu[i]
            imdict['image'] = image
            imdict['thumb'] = thumbs[i] if thumbs else None
            images.append(imdict)
    elif test and not imagelist:
        for i in range(num):
            imdict = defaultdict(str)
            imdict['name'] = '4444-0000'
            imdict['image'] = 'http://placehold.it/470x480&text={0}'.format(i)
            imdict['thumb'] = 'http://placehold.it/150x150&text={0}'.format(i)
            images.append(imdict)

    return images
