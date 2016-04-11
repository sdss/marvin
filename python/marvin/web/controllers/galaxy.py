#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-04-08 14:31:34
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-04-08 14:31:34 by Brian Cherinka
    Last Modified On: 2016-04-08 14:31:34 by Brian

'''
from __future__ import print_function
from __future__ import division
from flask import current_app, Blueprint, render_template, session as current_session, request, redirect, url_for, jsonify
from flask.ext.classy import FlaskView, route
from marvin.api.base import processRequest
from marvin.utils.general.general import convertIvarToErr, findClosestVector, convertImgCoords, isPlateifuOrMangaid as isPlateifuOrMangaid
from marvin.tools.core import MarvinError
from marvin.tools.cube import Cube

galaxy = Blueprint("galaxy_page", __name__)


def getWebSpectrum(cube, x, y, xyorig=None, byradec=False):
    ''' Get and format a spectrum for the web '''
    webspec = None
    try:
        if byradec:
            spectrum = cube.getSpectrum(ra=x, dec=y, xyorig=xyorig)
        else:
            spectrum = cube.getSpectrum(x=x, y=y, xyorig=xyorig)
    except Exception as e:
        specmsg = 'Could not get spectrum: {0}'.format(e)
    else:
        # get error and wavelength
        if byradec:
            ivar = cube.getSpectrum(ra=x, dec=y, ext='ivar', xyorig=xyorig)
        else:
            ivar = cube.getSpectrum(x=x, y=y, ext='ivar', xyorig=xyorig)
        error = convertIvarToErr(ivar)
        wave = cube.getWavelength()
        # make input array for Dygraph
        webspec = [[wave[i], [s, error[i]]] for i, s in enumerate(spectrum)]

        specmsg = "for relative coords x = {0}, y={1}".format(x, y)

    return webspec, specmsg


class Galaxy(FlaskView):
    route_base = '/galaxy'

    def __init__(self):
        ''' Initialize the route '''
        self.galaxy = {}
        self.galaxy['title'] = 'Marvin | Galaxy'
        self.galaxy['error'] = None

    def before_request(self, *args, **kwargs):
        ''' Do these things before a request to any route '''
        self.galaxy['error'] = None
        self.galaxy['cube'] = None
        self.galaxy['image'] = ''
        self.galaxy['spectra'] = 'null'

    def index(self):
        ''' Main galaxy page '''
        self.galaxy['error'] = 'Not all there are you...  Try adding a plate-IFU or manga-ID to the end of the address.'
        return render_template("galaxy.html", **self.galaxy)

    def get(self, galid):
        ''' Retrieve info for a given cube '''

        # determine type of galid
        self.galaxy['id'] = galid
        isvalid, idtype = isPlateifuOrMangaid(galid)
        if not isvalid:
            self.galaxy['error'] = 'Error: Galaxy ID {0} must either be a Plate-IFU, or MaNGA-Id designation.'.format(galid)
            return render_template("galaxy.html", **self.galaxy)
        else:
            # set plateifu or mangaid
            self.galaxy['idtype'] = idtype
            galaxyid = {self.galaxy['idtype']: galid}

            # Get cube
            try:
                cube = Cube(**galaxyid)
            except MarvinError as e:
                self.galaxy['cube'] = None
                self.galaxy['error'] = 'MarvinError: {0}'.format(e)
                return render_template("galaxy.html", **self.galaxy)
            else:
                self.galaxy['cube'] = cube
                self.galaxy['image'] = cube._cube.image

            # Get the initial spectrum
            if cube:
                webspec, specmsg = getWebSpectrum(cube, cube.ra, cube.dec, byradec=True)
                self.galaxy['spectra'] = webspec
                self.galaxy['specmsg'] = specmsg
                print(specmsg)

        return render_template("galaxy.html", **self.galaxy)

    @route('getspaxel', methods=['POST'], endpoint='getspaxel')
    def getSpaxel(self):
        f = processRequest(request=request)
        print('req', request.form)
        # for now, do this, but TODO - general processRequest to handle lists and not lists
        try:
            mousecoords = [float(v) for v in f.get('mousecoords[]', None)]
        except:
            mousecoords = None
        if mousecoords:
            print('form', f, mousecoords)
            arrshape = (34, 34)
            pixshape = (int(f['imwidth']), int(f['imheight']))
            if (mousecoords[0] < 0 or mousecoords[0] > pixshape[0]) or (mousecoords[1] < 0 or mousecoords[1] > pixshape[1]):
                output = {'message': 'error: pixel coords outside range'}
            else:
                xyorig = 'relative'
                arrcoords = findClosestVector(mousecoords, arr_shape=arrshape, pixel_shape=pixshape, xyorig=xyorig)
                oldarr = arrcoords
                infile = '/Users/Brian/Work/Manga/redux/v1_5_1/8485/stack/images/test_wcs2.png'
                arrcoords = convertImgCoords(mousecoords, infile, to_radec=True)
                print('arrcoords', oldarr, arrcoords)
                cube = Cube(plateifu=f['plateifu'])
                webspec, specmsg = getWebSpectrum(cube, arrcoords[0], arrcoords[1], xyorig=xyorig, byradec=True)
                print('specmsg', specmsg)
                msg = 'gettin some spaxel with {0} array coords (x,y): {1} at RA/Dec {2}'.format(xyorig, oldarr, arrcoords)
                output = {'message': msg, 'specmsg': specmsg, 'spectra': webspec}
        else:
            output = {'message': 'error getting mouse coords'}
        return jsonify(result=output)

Galaxy.register(galaxy)
