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
from brain.api.base import processRequest
from marvin.utils.general.general import findClosestVector, convertImgCoords, parseIdentifier
from brain.utils.general.general import convertIvarToErr
from marvin.core import MarvinError
from marvin.tools.cube import Cube
from marvin import config
import os

try:
    from sdss_access.path import Path
except ImportError as e:
    Path = None

galaxy = Blueprint("galaxy_page", __name__)


def getWebSpectrum(cube, x, y, xyorig=None, byradec=False):
    ''' Get and format a spectrum for the web '''
    webspec = None
    try:
        if byradec:
            spectrum = cube.getSpaxel(ra=x, dec=y, xyorig=xyorig)
        else:
            spectrum = cube.getSpaxel(x=x, y=y, xyorig=xyorig)
    except Exception as e:
        specmsg = 'Could not get spectrum: {0}'.format(e)
    else:
        # get error and wavelength
        error = convertIvarToErr(spectrum.ivar)
        wave = spectrum.wavelength
        # make input array for Dygraph
        webspec = [[wave[i], [s, error[i]]] for i, s in enumerate(spectrum.flux)]

        specmsg = "Spectrum in Spaxel ({2},{3}) at RA, Dec = ({0}, {1})".format(x, y, spectrum.x, spectrum.y)

    return webspec, specmsg


class Galaxy(FlaskView):
    route_base = '/galaxy'

    def __init__(self):
        ''' Initialize the route '''
        self.galaxy = {}
        self.galaxy['title'] = 'Marvin | Galaxy'
        self.galaxy['page'] = 'marvin-galaxy'
        self.galaxy['error'] = None
        self.galaxy['specmsg'] = None

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
        idtype = parseIdentifier(galid)
        if idtype in ['plateifu', 'mangaid']:
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
                # get SAS url links to cube, rss, maps, image
                if Path:
                    sdss_path = Path()
                    self.galaxy['image'] = sdss_path.url('mangaimage', drpver=cube._drpver, plate=cube.plate, ifu=cube.ifu, dir3d=cube.dir3d)
                    cubelink = sdss_path.url('mangacube', drpver=cube._drpver, plate=cube.plate, ifu=cube.ifu)
                    rsslink = sdss_path.url('mangarss', drpver=cube._drpver, plate=cube.plate, ifu=cube.ifu)
                    maplink = sdss_path.url('mangadefault', drpver=cube._drpver, dapver=cube._dapver, plate=cube.plate, ifu=cube.ifu)
                    self.galaxy['links'] = {'cube': cubelink, 'rss': rsslink, 'map': maplink}
                else:
                    self.galaxy['image'] = cube._cube.image
                print('image', cube._cube.image)

            # Get the initial spectrum
            if cube:
                webspec, specmsg = getWebSpectrum(cube, cube.ra, cube.dec, byradec=True)
                if not webspec:
                    self.galaxy['error'] = 'Error: {0}'.format(specmsg)
                self.galaxy['spectra'] = webspec
                self.galaxy['specmsg'] = specmsg
                self.galaxy['cubehdr'] = cube.hdr
                self.galaxy['quality'] = cube.qualitybit
                self.galaxy['mngtarget'] = cube.targetbit
        else:
            self.galaxy['error'] = 'Error: Galaxy ID {0} must either be a Plate-IFU, or MaNGA-Id designation.'.format(galid)
            return render_template("galaxy.html", **self.galaxy)

        return render_template("galaxy.html", **self.galaxy)

    @route('getspaxel', methods=['POST'], endpoint='getspaxel')
    def getSpaxel(self):
        f = processRequest(request=request)
        print(f)
        # for now, do this, but TODO - general processRequest to handle lists and not lists
        try:
            mousecoords = [float(v) for v in f.get('mousecoords[]', None)]
        except:
            mousecoords = None
        if mousecoords:
            pixshape = (int(f['imwidth']), int(f['imheight']))
            if (mousecoords[0] < 0 or mousecoords[0] > pixshape[0]) or (mousecoords[1] < 0 or mousecoords[1] > pixshape[1]):
                output = {'specmsg': 'Error: requested pixel coords are outside the image range.', 'status': -1}
                self.galaxy['error'] = output['specmsg']
            else:
                # TODO - generalize image file sas_url to filesystem switch, maybe in sdss_access
                infile = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'), f['image'].split('redux/')[1])
                arrcoords = convertImgCoords(mousecoords, infile, to_radec=True)
                cube = Cube(plateifu=f['plateifu'])
                webspec, specmsg = getWebSpectrum(cube, arrcoords[0], arrcoords[1], byradec=True)
                if not webspec:
                    self.galaxy['error'] = 'Error: {0}'.format(specmsg)
                    status = -1
                else:
                    status = 1
                msg = 'gettin some spaxel at RA/Dec {0}'.format(arrcoords)
                output = {'message': msg, 'specmsg': specmsg, 'spectra': webspec, 'status': status}
        else:
            output = {'specmsg': 'Error getting mouse coords', 'status': -1}
            self.galaxy['error'] = output['specmsg']

        return jsonify(result=output)

Galaxy.register(galaxy)
