from flask import current_app, Blueprint, render_template, jsonify
from flask import session as current_session, request, redirect, url_for
from flask_classy import FlaskView, route
from marvin import config, marvindb
from brain.api.base import processRequest
from marvin.utils.general.general import parseIdentifier
from marvin.web.web_utils import parseSession
import json
from hashlib import md5
try:
    from inspection.marvin import Inspection
except:
    from brain.core.inspection import Inspection

index = Blueprint("index_page", __name__)


class Marvin(FlaskView):
    route_base = '/'

    def __init__(self):
        self.base = {}
        self.base['title'] = 'Marvin'
        self.base['intro'] = 'Welcome to Marvin!'
        self.base['page'] = 'marvin-main'

    def index(self):
        current_app.logger.info('Welcome to Marvin Web!')

        return render_template("index.html", **self.base)

    def quote(self):
        return 'getting quote'

    @route('/test/')
    def test(self):
        return 'new test'

    def database(self):
        onecube = marvindb.session.query(marvindb.datadb.Cube).first()
        return str(onecube.plate)

    @route('/galidselect/', methods=['GET', 'POST'], endpoint='galidselect')
    def galidselect(self):
        ''' Route that handle the Navbar plate/galaxy id search form '''
        f = processRequest(request=request)
        galid = f['galid']
        idtype = parseIdentifier(galid)
        if idtype == 'plateifu' or idtype == 'mangaid':
            return redirect(url_for('galaxy_page.Galaxy:get', galid=galid))
        elif idtype == 'plate':
            return redirect(url_for('plate_page.Plate:get', plateid=galid))
        else:
            return redirect(url_for('index_page.Marvin:index'))

    @route('/getgalidlist/', methods=['GET', 'POST'], endpoint='getgalidlist')
    def getgalidlist(self):
        ''' Retrieves the list of galaxy ids and plates for Bloodhound Typeahead '''
        self._drpver, self._dapver, self._release = parseSession()
        if marvindb.datadb is None:
            out = ['', '', '']
            current_app.logger.info('ERROR: Problem with marvindb.datadb.  Cannot build galaxy id auto complete list.')
        else:
            cubes = (marvindb.session.query(marvindb.datadb.Cube.plate, marvindb.datadb.Cube.mangaid,
                                            marvindb.datadb.Cube.plateifu).join(marvindb.datadb.PipelineInfo,
                                                                                marvindb.datadb.PipelineVersion,
                                                                                marvindb.datadb.IFUDesign).
                     filter(marvindb.datadb.PipelineVersion.version == self._drpver).all())
            out = [str(e) for l in cubes for e in l]
        out = list(set(out))
        out.sort()
        return json.dumps(out)

    @route('/selectmpl/', methods=['GET', 'POST'], endpoint='selectmpl')
    def selectmpl(self):
        ''' Global selection of the MPL/DR versions '''
        f = processRequest(request=request)
        out = {'status': 1, 'msg': 'Success'}
        version = f['mplselect']
        print('setting new mpl', version)
        current_session['currentver'] = version
        drpver, dapver = config.lookUpVersions(release=version)
        current_session['drpver'] = drpver
        current_session['dapver'] = dapver

        return jsonify(result=out)

    @route('/login/', methods=['GET', 'POST'], endpoint='login')
    def login(self):
        ''' login for trac user '''
        form = processRequest(request=request)
        result = {}
        username = form['username']
        password = form['password']
        auth = md5("%s:AS3Trac:%s" % (username.strip(), password.strip())).hexdigest() if username and password else None
        try:
            inspection = Inspection(current_session, username=username, auth=auth)
        except Exception as e:
            result['status'] = -1
            result['message'] = e
            current_session['loginready'] = False
        else:
            result = inspection.result()
            current_session['loginready'] = inspection.ready
            current_session['name'] = result['membername']
        print('login result', result)
        return jsonify(result=result)

Marvin.register(index)
