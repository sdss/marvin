from flask import current_app, Blueprint, render_template, session as current_session, request, redirect, url_for, jsonify
from flask.ext.classy import FlaskView, route
from marvin import config, marvindb
from marvin.tools.query.forms import MarvinForm
from brain.api.base import processRequest
from marvin.tools.query import Query, doQuery
from marvin.core import MarvinError
from wtforms import SelectField, validators
import json

index = Blueprint("index_page", __name__)


class Marvin(FlaskView):
    route_base = '/'

    def index(self):
        index = {}
        index['title'] = 'Marvin'
        index['intro'] = 'Welcome to Marvin'
        config.drpver = 'v1_5_1'
        mangaid = '1-209232'

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


Marvin.register(index)
