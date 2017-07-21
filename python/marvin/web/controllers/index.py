from flask import current_app, Blueprint, render_template, jsonify
from flask import session as current_session, request, redirect, url_for
from flask_classy import route
from marvin import config, marvindb
from brain.api.base import processRequest
from marvin.utils.general.general import parseIdentifier
from marvin.web.web_utils import parseSession
from marvin.api.base import arg_validate as av
from marvin.web.controllers import BaseWebView

from hashlib import md5
# try:
#     print('importing main inspection')
#     from inspection.marvin import Inspection
# except ImportError as e:
#     print('importing local inspection')
#     from brain.core.inspection import Inspection

index = Blueprint("index_page", __name__)


class Marvin(BaseWebView):
    route_base = '/'

    def __init__(self):
        super(Marvin, self).__init__('marvin-main')
        self.main = self.base.copy()

    def before_request(self, *args, **kwargs):
        super(Marvin, self).before_request(*args, **kwargs)
        self.reset_dict(self.main)

    def index(self):
        current_app.logger.info('Welcome to Marvin Web!')

        return render_template("index.html", **self.main)

    def quote(self):
        return 'getting quote'

    def status(self):
        return 'OK'

    @route('/test/')
    def test(self):
        return 'new test'

    def database(self):
        onecube = marvindb.session.query(marvindb.datadb.Cube).order_by(marvindb.datadb.Cube.pk).first()
        return jsonify(result={'plate': onecube.plate, 'status': 1})

    @route('/galidselect/', methods=['GET', 'POST'], endpoint='galidselect')
    def galidselect(self):
        ''' Route that handle the Navbar plate/galaxy id search form '''
        args = av.manual_parse(self, request, use_params='index', required='galid')
        galid = args.get('galid', None)
        if not galid:
            # if not galid return main page
            return redirect(url_for('index_page.Marvin:index'))
        else:
            idtype = parseIdentifier(galid)
        # check the idtype
        if idtype == 'plateifu' or idtype == 'mangaid':
            return redirect(url_for('galaxy_page.Galaxy:get', galid=galid))
        elif idtype == 'plate':
            return redirect(url_for('plate_page.Plate:get', plateid=galid))
        else:
            return redirect(url_for('index_page.Marvin:index'))

    @route('/getgalidlist/', methods=['GET', 'POST'], endpoint='getgalidlist')
    def getgalidlist(self):
        ''' Retrieves the list of galaxy ids and plates for Bloodhound Typeahead '''

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
        return jsonify(out)

    @route('/selectmpl/', methods=['GET', 'POST'], endpoint='selectmpl')
    def selectmpl(self):
        ''' Global selection of the MPL/DR versions '''
        args = av.manual_parse(self, request, use_params='index')
        version = args.get('release', None)
        current_session['release'] = version
        drpver, dapver = config.lookUpVersions(release=version)
        current_session['drpver'] = drpver
        current_session['dapver'] = dapver
        out = {'status': 1, 'msg': 'Success', 'current_release': version,
               'current_drpver': drpver, 'current_dapver': dapver}

        return jsonify(result=out)

    @route('/login/', methods=['GET', 'POST'], endpoint='login')
    def login(self):
        ''' login for trac user '''
        try:
            from inspection.marvin import Inspection
        except ImportError as e:
            from brain.core.inspection import Inspection

        form = processRequest(request=request)
        result = {}
        username = form['username'].strip()
        password = form['password'].strip()
        auth = md5("{0}:AS3Trac:{1}".format(username, password).encode('utf-8')).hexdigest() if username and password else None
        try:
            inspection = Inspection(current_session, username=username, auth=auth)
        except Exception as e:
            result['status'] = -1
            result['message'] = e
            current_session['loginready'] = False
        else:
            result = inspection.result()
            current_session['loginready'] = inspection.ready
            current_session['name'] = result.get('membername', None)
        print('login result', result)
        return jsonify(result=result)

Marvin.register(index)
