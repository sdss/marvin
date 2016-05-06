from flask import current_app, Blueprint, render_template, session as current_session, request, redirect, url_for, jsonify
from flask.ext.classy import FlaskView, route
from marvin import config, marvindb
from brain.api.base import processRequest
from marvin.core import MarvinError
from marvin.utils.general.general import parseIdentifier
from wtforms import SelectField, validators
import json

index = Blueprint("index_page", __name__)


class Marvin(FlaskView):
    route_base = '/'

    def __init__(self):
        self.base = {}
        self.base['title'] = 'Marvin'
        self.base['intro'] = 'Welcome to Marvin!'
        self.base['page'] = 'marvin-main'

    def index(self):
        config.drpver = 'v1_5_1'
        mangaid = '1-209232'
        current_app.logger.info('Welcome to Marvin Web!')

        # get all MPLs
        mpls = config._mpldict.keys()
        versions = [{'name': mpl, 'subtext': str(config.lookUpVersions(mpl))} for mpl in mpls]
        current_session['versions'] = versions

        # set default - TODO replace with setGlobalSession
        if 'currentmpl' not in current_session:
            current_session['currentmpl'] = config.mplver

        return render_template("index.html", **self.base)

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
        cubes = marvindb.session.query(marvindb.datadb.Cube.plate, marvindb.datadb.Cube.mangaid,
                                       marvindb.datadb.Cube.plateifu).join(marvindb.datadb.PipelineInfo,
                                                                           marvindb.datadb.PipelineVersion,
                                                                           marvindb.datadb.IFUDesign).\
            filter(marvindb.datadb.PipelineVersion.version == config.drpver).all()
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
        current_session['currentmpl'] = version
        if 'MPL' in version:
            config.setMPL(version)
        elif 'DR' in version:
            config.setDR(version)
        else:
            out['status'] = -1
            out['msg'] = 'version {0} is neither an MPL or DR'.format(version)

        return jsonify(result=out)


Marvin.register(index)
