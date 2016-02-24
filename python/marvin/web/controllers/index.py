from flask import current_app, Blueprint, render_template, session as current_session, request
from flask.ext.classy import FlaskView, route
from marvin.tools.cube import Cube
from marvin import config, session, datadb
from marvin.utils.general.general import convertIvarToErr
import numpy as np
from marvin.tools.query.forms import TestForm, SampleForm
from marvin.api.base import processRequest
from sqlalchemy import or_, and_
from marvin.tools.query import Query

index = Blueprint("index_page", __name__)


class Marvin(FlaskView):
    route_base = '/'

    def index(self):
        index = {}
        index['title'] = 'Marvin'
        index['intro'] = 'Welcome to Marvin'
        config.drpver = 'v1_5_1'
        cube = Cube(mangaid='1-209232')
        index['cube'] = cube
        x = 1
        y = 2
        try:
            spectrum = cube.getSpectrum(x=x, y=y)
        except Exception as e:
            spectrum = 'Could not get spectrum: {0}'.format(e)
        # get error and wavelength
        ivar = cube.getSpectrum(x=x, y=y, ext='ivar')
        error = convertIvarToErr(ivar)
        wave = cube.getWavelength()
        # make input array for Dygraph
        webspec = [[wave[i], [s, error[i]]] for i, s in enumerate(spectrum)]

        index['specmsg'] = "for x = {0}, y={1}".format(x, y)
        index['spectra'] = webspec

        t = TestForm()
        index['testform'] = t
        s = SampleForm()
        index['sampleform'] = s
        return render_template("index.html", **index)

    def quote(self):
        return 'getting quote'

    def get(self, id):
        return 'getting id {0}'.format(id)

    @route('/test/')
    def test(self):
        return 'new test'

    def database(self):
        onecube = session.query(datadb.Cube).first()
        return str(onecube.plate)

    @route('/search', methods=['POST'])
    def search(self):
        print('here i am searching')
        f = processRequest(request=request)
        print(f)
        print('form', request.form)
        # s = SampleForm(request.form)
        # print('s', s.data)
        # res = session.query(datadb.Cube).join(s.Meta.model)
        # f = None
        # for key, val in s.data.items():
        #     if val:
        #         if not f:
        #             f = and_(s.Meta.model.__table__.columns.__getitem__(key) < val)
        #         else:
        #             f = and_(f, s.Meta.model.__table__.columns.__getitem__(key) < val)
        # res = res.filter(f).all()

        # testing the rough query version of the above
        q = Query()
        q._createBaseQuery()
        q.set_params(params=f)
        q.add_condition()
        res = q.run()

        #t = TestForm()
        #print('t', t.data)
        test = {'results': len(res.results)}
        return render_template('test.html', **test)

Marvin.register(index)
