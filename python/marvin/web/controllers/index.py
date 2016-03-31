from flask import current_app, Blueprint, render_template, session as current_session, request, redirect, url_for, jsonify
from flask.ext.classy import FlaskView, route
from marvin.tools.cube import Cube
from marvin import config, marvindb
from marvin.utils.general.general import convertIvarToErr, findClosestVector
import numpy as np
from marvin.tools.query.forms import MarvinForm
from marvin.api.base import processRequest
from sqlalchemy import or_, and_
from marvin.tools.query import Query, doQuery
from marvin.tools.core import MarvinError
from wtforms import SelectField, validators
import json

index = Blueprint("index_page", __name__)

opdict = {'le': '<=', 'ge': '>=', 'gt': '>', 'lt': '<', 'ne': '!=', 'eq': '='}
ops = [(key, val) for key, val in opdict.items()]


def getWebSpectrum(cube, x, y, xyorig=None):
    ''' get and format a spectrum for the web '''
    webspec = None
    try:
        spectrum = cube.getSpectrum(x=x, y=y, xyorig=xyorig)
    except Exception as e:
        specmsg = 'Could not get spectrum: {0}'.format(e)
    else:
        # get error and wavelength
        ivar = cube.getSpectrum(x=x, y=y, ext='ivar', xyorig=xyorig)
        error = convertIvarToErr(ivar)
        wave = cube.getWavelength()
        # make input array for Dygraph
        webspec = [[wave[i], [s, error[i]]] for i, s in enumerate(spectrum)]

        specmsg = "for relative coords x = {0}, y={1}".format(x, y)

    return webspec, specmsg


class Marvin(FlaskView):
    route_base = '/'

    def index(self):
        index = {}
        index['title'] = 'Marvin'
        index['intro'] = 'Welcome to Marvin'
        config.drpver = 'v1_5_1'
        mangaid = '1-209232'
        try:
            cube = Cube(mangaid=mangaid)
        except MarvinError as e:
            cube = None
        index['cube'] = cube
        index['mangaid'] = mangaid
        index['plateifu'] = cube.plateifu
        x = 1
        y = 2
        if cube:
            webspec, specmsg = getWebSpectrum(cube, x, y)
            index['spectra'] = webspec
            index['specmsg'] = specmsg

        # general marvin form
        m = MarvinForm()
        # build new ifu select and add it to the IFUDesignForm
        _ifus = sorted(list(set([i.name[:-2] for i in marvindb.session.query(marvindb.datadb.IFUDesign).all()])), key=lambda t: int(t))
        _ifufields = [('{0}'.format(_i), _i) for _i in _ifus]
        ifu = SelectField('IFU Design', choices=_ifufields)
        m._param_form_lookup['ifu.name'].name = ifu

        # generate ifu and sample form fields
        ifuform = m.callInstance(m._param_form_lookup['ifu.name'])
        sampform = m.callInstance(m._param_form_lookup['nsa_redshift'], validators=[validators.regexp('([0-9])+')])
        mainform = m.MainForm()

        # pass into template dictionary
        index['ifuform'] = ifuform
        index['sampleform'] = sampform
        index['mainform'] = mainform
        return render_template("index.html", **index)

    def quote(self):
        return 'getting quote'

    def get(self, id):
        return 'getting id {0}'.format(id)

    @route('/test/')
    def test(self):
        return 'new test'

    def database(self):
        onecube = marvindb.session.query(datadb.Cube).first()
        return str(onecube.plate)

    @route('/search', methods=['POST'])
    def search(self):
        print('here i am searching')
        f = processRequest(request=request)
        print('params', f)
        print('form', request.form)
        test = {'results': None, 'errmsg': None}

        m = MarvinForm()
        mainform = m.MainForm(**f)
        test['mainform'] = mainform
        searchvalue = f['searchbox']
        current_session.update({'searchvalue': searchvalue})
        if mainform.validate():
            print('form validated, doing query')
            # testing the rough query version of the above
            try:
                q, res = doQuery(searchvalue)
            except MarvinError as e:
                test['errmsg'] = 'Could not perform query: {0}'.format(e)
            else:
                test['filter'] = q.strfilter
                test['count'] = res.count
                test['results'] = len(res.results)

        return render_template('test.html', **test)

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
            pixshape = (252, 252)
            if (mousecoords[0] < 0 or mousecoords[0] > 252) or (mousecoords[1] < 0 or mousecoords[1] > 252):
                output = {'message': 'error: pixel coords outside range'}
            else:
                xyorig = 'relative'
                arrcoords = findClosestVector(mousecoords, arr_shape=arrshape, pixel_shape=pixshape, xyorig=xyorig)
                cube = Cube(plateifu=f['plateifu'])
                webspec, specmsg = getWebSpectrum(cube, arrcoords[0], arrcoords[1], xyorig=xyorig)
                print('specmsg', specmsg)
                msg = 'gettin some spaxel with {0} array coords (x,y): {1}'.format(xyorig, arrcoords)
                output = {'message': msg, 'specmsg': specmsg, 'spectra': webspec}
        else:
            output = {'message': 'error getting mouse coords'}
        return jsonify(result=output)

Marvin.register(index)

