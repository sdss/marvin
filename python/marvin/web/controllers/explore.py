# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Filename: explore.py
# Project: controllers
# Author: Brian Cherinka
# Created: Tuesday, 7th July 2020 3:45:32 pm
# License: BSD 3-clause "New" or "Revised" License
# Copyright (c) 2020 Brian Cherinka
# Last Modified: Tuesday, 7th July 2020 3:45:32 pm
# Modified By: Brian Cherinka


from __future__ import print_function, division, absolute_import

from flask import Blueprint, render_template, request, redirect, after_this_request
from flask import url_for, flash, current_app, session as current_session, jsonify
from flask_classful import route
from marvin.core.exceptions import MarvinError
from marvin.web.controllers import BaseWebView
from marvin.utils.datamodel.dap import datamodel
from marvin.utils.general import parseIdentifier
from werkzeug.utils import secure_filename
import os
import pandas as pd
from brain.api.base import processRequest

from marvin.api.base import arg_validate as av
from marvin.web.controllers.galaxy import buildMapDict
from marvin.tools import Cube

explore = Blueprint("explore_page", __name__)


def allowed_file(filename, extensions):
    ''' check if the filename has an allowed extension '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extensions


def read_targets(filename):
    ''' read the list of targets from a filename '''
    try:
        df = pd.read_csv(filename)
    except Exception:
        targets = []
    else:
        # get the first column
        targets = df[df.columns[0]].astype(str).tolist()
    return targets


def check_targets(targets):
    ''' check the target string is either plateifu or mangaid '''
    return [t for t in targets if parseIdentifier(t) in ['plateifu', 'mangaid']]

class Explore(BaseWebView):
    route_base = '/explore/'

    def __init__(self):
        ''' Initialize the route '''
        super(Explore, self).__init__('marvin-explore')
        self.explore = self.base.copy()
        self.explore['n_targs'] = 0
        self.explore['targetlist'] = ''
        self.explore['uploaded'] = False
        self.explore['targets'] = None
        self.explore['mapchoice'] = 'emline_gflux:ha_6564'
        self.explore['maps'] = None
        self.explore['mapmsgs'] = None
        self.explore['mapsloading'] = False

    def before_request(self, *args, **kwargs):
        ''' Do these things before a request to any route '''
        super(Explore, self).before_request(*args, **kwargs)
        self.reset_dict(self.explore, exclude=['n_targs', 'mapchoice'])

    @route('/', methods=['GET', 'POST'])
    def index(self):

        dm = datamodel[self._dapver]
        daplist = [p.full(web=True) for p in dm.properties]

        self.explore['dapbintemps'] = dm.get_bintemps(db_only=True)
        self.explore['dapmaps'] = daplist

        if 'bintemp' not in current_session:
            current_session['bintemp'] = '{0}-{1}'.format(dm.get_bintype(), dm.get_template())

        self.explore['targetlist'] = current_session.get('targetlist', '')
        self.explore['n_targs'] = current_session.get('n_targs', 0)

        return render_template('explore.html', **self.explore)

    @route('/upload/', methods=['GET', 'POST'], endpoint='upload')
    def upload_file(self):
        home = 'explore_page.Explore:index'
        if request.method == 'GET':
            flash('Method not allowed', 'error')
            return redirect(url_for(home))

        # no file part
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for(home))

        # empty file
        fileobj = request.files['file']
        if fileobj.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for(home))

        # bad extension
        extensions = current_app.config['ALLOWED_EXTENSIONS']
        if not allowed_file(fileobj.filename, extensions):
            flash('File must be one of extension: {0}'.format(list(extensions)), 'error')
            return redirect(url_for(home))

        # save the file
        filename = secure_filename(fileobj.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        fileobj.save(filepath)

        # delete the file after the request
        @after_this_request
        def remove_file(response):
            if os.path.exists(filepath):
                os.remove(filepath)
            return response

        # bad size
        filesize = os.path.getsize(filepath)
        if filesize > current_app.config['MAX_CONTENT_LENGTH']:
            flash('Uploaded file is too large', 'error')
            return redirect(url_for(home)), 413

        # read targets
        targets = read_targets(filepath)
        if not targets:
            flash('Could not read targets from file.  Check file format.', 'error')
            return redirect(url_for(home))
        else:
            orig_count = len(targets)
            targets = check_targets(targets)
            if not targets:
                flash('No valid targets found in format of plateifu or mangaid', 'error')
                return redirect(url_for(home))

            # check number of targets
            n_targets = len(targets)

            if n_targets < orig_count:
                flash(f'Removed {orig_count - n_targets} invalid targets from list', 'warning')

            msg = 'Targets successfully uploaded.'
            if n_targets > 100:
                targets = targets[:100]
                msg += ' Only a limit of 100 targets allowed.'
            msg += ' Now select a map below.'

            flash(msg, 'info')
            targetlist = '&#10;'.join(targets)

            dm = datamodel[self._dapver]
            daplist = [p.full(web=True) for p in dm.properties]
            self.explore['dapmaps'] = daplist
            self.explore['dapbintemps'] = dm.get_bintemps(db_only=True)
            self.explore['targets'] = targets
            self.explore['targetlist'] = targetlist
            self.explore['n_targs'] = len(targets)
            return render_template('explore.html', **self.explore)

    @route('/maps/', methods=['GET', 'POST'], endpoint='maps')
    def get_maps(self):
        stuff = processRequest(request)
        home = 'explore_page.Explore:index'

        dm = datamodel[self._dapver]
        daplist = [p.full(web=True) for p in dm.properties]

        self.explore['dapmaps'] = daplist
        self.explore['dapbintemps'] = dm.get_bintemps(db_only=True)

        targetlist = stuff.get('targetlist', '')
        mapchoice = stuff.get('mapchoice', '')
        btchoice = stuff.get('btchoice', '')

        if not targetlist:
            flash('Must provide a list of targets', 'error')
            return redirect(url_for(home))

        if not mapchoice or not btchoice:
            flash('Must select a map and bintype', 'error')
            return redirect(url_for(home))
        
        # update the DAPTYPE session selection
        current_session['bintemp'] = btchoice

        targets = targetlist.split('\r\n')
        self.explore['targetlist'] = targetlist
        self.explore['targets'] = targets
        self.explore['n_targs'] = len(targets)
        self.explore['mapchoice'] = mapchoice
        self.explore['btchoice'] = btchoice

        self.explore['maps'] = True
        self.explore['mapmsgs'] = 'go'
        self.explore['mapsloading'] = True

        return render_template('explore.html', **self.explore)
    
    @route('/webmap/', methods=['POST'], endpoint='webmap')
    def updateMaps(self):
        args = av.manual_parse(self, request, use_params='explore', required='target')
        #stuff = processRequest(request)
        mapchoice = args.get('mapchoice', '')
        btchoice = args.get('btchoice', '')
        target = args.get('target', '')

        output = {'maps': None, 'msg': None}

        try:
            cube = Cube(target, release=self._release)
        except MarvinError:
            mapdict = {'data': None, 'msg': 'Error',
                        'plotparams': None}
            mapmsg = 'No cube available'
            output['maps'] = mapdict
            output['msg'] = mapmsg
            return jsonify(result=output)

        hasbin = btchoice.split('-')[0] in cube.get_available_bintypes() if btchoice else None
        # in principle update the DAPTYPE session selection, but doesn't quite work
        current_session['bintemp'] = btchoice

        try:
            mapdict = buildMapDict(cube, [mapchoice], self._dapver, bintemp=btchoice)
            mapmsg = None
        except Exception as e:
            mapdict = {'data': None, 'msg': 'Error', 'plotparams': None}
            if hasbin:
                mapmsg = 'Error getting maps: {0}'.format(e)
            else:
                mapmsg = 'No maps available for selected bintype {0}. Try a different one.'.format(
                    btchoice)
            output['maps'] = mapdict
            output['msg'] = mapmsg
            return jsonify(result=output)
        else:
            self.explore['mapstatus'] = 1
            output['maps'] = mapdict[0]
            output['msg'] = mapmsg
            return jsonify(result=output)
        

Explore.register(explore)
