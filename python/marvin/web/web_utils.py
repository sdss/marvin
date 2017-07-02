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
from flask import session as current_session, render_template, request, g, jsonify
from marvin import config
from marvin.utils.db import get_traceback
from collections import defaultdict
import re
import traceback


def configFeatures(app, mode):
    ''' Configure Flask Feature Flags '''

    app.config['FEATURE_FLAGS']['collab'] = False if mode == 'dr13' else True
    app.config['FEATURE_FLAGS']['new'] = False if mode == 'dr13' else True
    app.config['FEATURE_FLAGS']['dev'] = True if config.db == 'local' else False


def updateGlobalSession():
    ''' updates the Marvin config with the global Flask session '''

    # check if mpl versions in session
    if 'versions' not in current_session:
        setGlobalSession()
    elif 'drpver' not in current_session or \
         'dapver' not in current_session:
        drpver, dapver = config.lookUpVersions(release=config.release)
        current_session['drpver'] = drpver
        current_session['dapver'] = dapver
    elif 'release' not in current_session:
        current_session['release'] = config.release


def setGlobalSession():
    ''' Sets the global session for Flask '''

    mpls = list(config._mpldict.keys())
    versions = [{'name': mpl, 'subtext': str(config.lookUpVersions(release=mpl))} for mpl in mpls]
    current_session['versions'] = versions

    if 'release' not in current_session:
        current_session['release'] = config.release
        drpver, dapver = config.lookUpVersions(release=config.release)
        current_session['drpver'] = drpver
        current_session['dapver'] = dapver


def parseSession():
    ''' parse the current session for MPL versions '''
    drpver = current_session['drpver']
    dapver = current_session['dapver']
    release = current_session['release']
    return drpver, dapver, release


# def make_error_json(error, name, code):
#     ''' creates the error json dictionary for API errors '''
#     shortname = name.lower().replace(' ', '_')
#     messages = {'error': shortname,
#                 'message': error.description if hasattr(error, 'description') else None,
#                 'status_code': code,
#                 'traceback': get_traceback(asstring=True)}
#     return jsonify({'api_error': messages}), code


# def make_error_page(app, name, code, sentry=None, data=None):
#     ''' creates the error page dictionary for web errors '''
#     shortname = name.lower().replace(' ', '_')
#     error = {}
#     error['title'] = 'Marvin | {0}'.format(name)
#     error['page'] = request.url
#     error['event_id'] = g.get('sentry_event_id', None)
#     error['data'] = data
#     if sentry:
#         error['public_dsn'] = sentry.client.get_public_dsn('https')
#     app.logger.error('{0} Exception {1}'.format(name, error))
#     return render_template('errors/{0}.html'.format(shortname), **error), code


def send_request():
    ''' sends the request info to the history db '''
    if request.blueprint is not None:
        print(request.cookies)
        print(request.headers)
        print(request.blueprint)
        print(request.endpoint)
        print(request.url)
        print(request.remote_addr)
        print(request.environ)
        print(request.environ['REMOTE_ADDR'])

# def setGlobalSession_old():
#     ''' Set default global session variables '''

#     if 'currentver' not in current_session:
#         setGlobalVersion()
#     if 'searchmode' not in current_session:
#         current_session['searchmode'] = 'plateid'
#     if 'marvinmode' not in current_session:
#         current_session['marvinmode'] = 'mangawork'
#     configFeatures(current_app, current_session['marvinmode'])
#     # current_session['searchoptions'] = getDblist(current_session['searchmode'])

#     # get code versions
#     #if 'codeversions' not in current_session:
#     #    buildCodeVersions()

#     # user authentication
#     if 'http_authorization' not in current_session:
#         try:
#             current_session['http_authorization'] = request.environ['HTTP_AUTHORIZATION']
#         except:
#             pass


# def getDRPVersion():
#     ''' Get DRP version to use during MaNGA SAS '''

#     # DRP versions
#     vers = marvindb.session.query(marvindb.datadb.PipelineVersion).\
#         filter(marvindb.datadb.PipelineVersion.version.like('%v%')).\
#         order_by(marvindb.datadb.PipelineVersion.version.desc()).all()
#     versions = [v.version for v in vers]

#     return versions


# def getDAPVersion():
#     ''' Get DAP version to use during MaNGA SAS '''

#     # DAP versions
#     vers = marvindb.session.query(marvindb.datadb.PipelineVersion).\
#         join(marvindb.datadb.PipelineInfo, marvindb.datadb.PipelineName).\
#         filter(marvindb.datadb.PipelineName.label == 'DAP',
#                ~marvindb.datadb.PipelineVersion.version.like('%trunk%')).\
#         order_by(marvindb.datadb.PipelineVersion.version.desc()).all()
#     versions = [v.version for v in vers]

#     return versions+['NA']


# def setGlobalVersion():
#     ''' set the global version '''

#     # set MPL version
#     try:
#         mplver = current_session['currentmpl']
#     except:
#         mplver = None
#     if not mplver:
#         current_session['currentmpl'] = 'MPL-4'

#     # set version mode
#     try:
#         vermode = current_session['vermode']
#     except:
#         vermode = None
#     if not vermode:
#         current_session['vermode'] = 'MPL'

#     # initialize
#     if 'MPL' in current_session['vermode']:
#         setMPLVersion(current_session['currentmpl'])

#     # set global DRP version
#     try:
#         versions = current_session['versions']
#     except:
#         versions = getDRPVersion()
#     current_session['versions'] = versions
#     try:
#         drpver = current_session['currentver']
#     except:
#         drpver = None
#     if not drpver:
#         realvers = [ver for ver in versions if os.path.isdir(os.path.join(os.getenv('MANGA_SPECTRO_REDUX'), ver))]
#         current_session['currentver'] = realvers[0]

#     # set global DAP version
#     try:
#         dapversions = current_session['dapversions']
#     except:
#         dapversions = getDAPVersion()
#     current_session['dapversions'] = dapversions
#     try:
#         ver = current_session['currentdapver']
#     except:
#         ver = None
#     if not ver:
#         realvers = [ver for ver in versions if os.path.isdir(os.path.join(os.getenv('MANGA_SPECTRO_ANALYSIS'),
#                     current_session['currentver'], ver))]
#         current_session['currentdapver'] = realvers[0] if realvers else 'NA'


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
        for i in xrange(num):
            imdict = defaultdict(str)
            imdict['name'] = '4444-0000'
            imdict['image'] = 'http://placehold.it/470x480&text={0}'.format(i)
            imdict['thumb'] = 'http://placehold.it/150x150&text={0}'.format(i)
            images.append(imdict)

    return images
