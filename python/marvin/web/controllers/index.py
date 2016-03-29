from flask import current_app, Blueprint, render_template, session as current_session, request, redirect, url_for
from flask.ext.classy import FlaskView, route
from marvin.tools.cube import Cube
from marvin import config, marvindb
from marvin.utils.general.general import convertIvarToErr
import numpy as np
from marvin.tools.query.forms import MarvinForm
from marvin.api.base import processRequest
from sqlalchemy import or_, and_
from marvin.tools.query import Query
from marvin.tools.core import MarvinError
from wtforms import SelectField, validators
import json

index = Blueprint("index_page", __name__)

opdict = {'le': '<=', 'ge': '>=', 'gt': '>', 'lt': '<', 'ne': '!=', 'eq': '='}
ops = [(key, val) for key, val in opdict.items()]


class Table(FlaskView):
    route_base = 'tables'

    @route('/getdata', methods=['GET', 'POST'], endpoint='getdata')
    def getData(self):
        print('inside get data')
        f = processRequest(request=request)
        print('getdata form', f)
        print('current_session', current_session['searchvalue'])
        searchvalue = current_session['searchvalue']
        limit = f['limit']
        offset = f['offset']
        order = f['order']
        q, res = doQuery(searchvalue, limit=limit)
        # sort
        #revorder = 'desc' in order
        #print('reverse', revorder, order)
        #res.results = sorted(res.results, key=lambda x: x.plateifu, reverse=revorder)
        # get subset on a given page
        results = res.getSubset(offset, limit=limit)
        rows = [{'plateifu': r.plateifu} for r in results]
        stuff = {'total': res.count, 'rows': rows}
        stuff = json.dumps(stuff)
        return stuff


def doQuery(searchvalue, limit=10):
    q = Query(limit=limit)
    q.set_filter(params=searchvalue)
    q.add_condition()
    res = q.run()
    return q, res


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
        x = 1
        y = 2
        if cube:
            try:
                spectrum = cube.getSpectrum(x=x, y=y)
            except Exception as e:
                spectrum = 'Could not get spectrum: {0}'.format(e)
            else:
                # get error and wavelength
                ivar = cube.getSpectrum(x=x, y=y, ext='ivar')
                error = convertIvarToErr(ivar)
                wave = cube.getWavelength()
                # make input array for Dygraph
                webspec = [[wave[i], [s, error[i]]] for i, s in enumerate(spectrum)]

                index['specmsg'] = "for x = {0}, y={1}".format(x, y)
                index['spectra'] = webspec

        # general marvin form
        m = MarvinForm()
        # build new ifu select and add it to the IFUDesignForm
        _ifus = sorted(list(set([i.name[:-2] for i in marvindb.session.query(marvindb.datadb.IFUDesign).all()])), key=lambda t: int(t))
        _ifufields = [('{0}'.format(_i), _i) for _i in _ifus]
        ifu = SelectField('IFU Design', choices=_ifufields)
        m._param_form_lookup['name'].name = ifu

        # generate ifu and sample form fields
        ifuform = m.callInstance(m._param_form_lookup['name'])
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

Marvin.register(index)
Table.register(index)

