from flask import current_app, Blueprint, render_template, jsonify
from flask import session as current_session, request, redirect, url_for
from flask_classful import route
from marvin import config
from brain.api.base import processRequest
from marvin.utils.general.general import parseIdentifier
from marvin.api.base import arg_validate as av
from marvin.web.controllers import BaseWebView
from marvin.web.web_utils import setGlobalSession, set_session_versions, get_web_releases
from marvin.web.extensions import cache

from brain.utils.general import validate_user, get_db_user
from flask_login import current_user, login_user, logout_user

if config.db:
    from marvin import marvindb

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

    @route('/versions/')
    def get_versions(self):
        vers = {'sess_vers': current_session['versions'], 'config_vers': list(get_web_releases().keys()),
                'access': config.access, 'release': config.release, 'session_release': current_session['release']}
        return jsonify(result=vers)

    @route('/session/')
    def get_session(self):
        extra = {'access': config.access}
        return jsonify(result=dict(current_session, **extra))

    @route('/clear/')
    def clear_session(self):
        # clear the session
        current_session.clear()
        # clear the cache
        cache.clear()
        return jsonify(result=dict(current_session))

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
                     filter(marvindb.datadb.PipelineVersion.version == str(self._drpver)).all())
            out = [str(e) for l in cubes for e in l]

        out = list(set(out))
        out.sort()
        return jsonify(out)

    @route('/selectmpl/', methods=['GET', 'POST'], endpoint='selectmpl')
    def selectmpl(self):
        ''' Global selection of the MPL/DR versions '''
        args = av.manual_parse(self, request, use_params='index')
        version = args.get('release', None)
        set_session_versions(version)
        drpver, dapver = config.lookUpVersions(release=version)
        out = {'status': 1, 'msg': 'Success', 'current_release': version,
               'current_drpver': drpver, 'current_dapver': dapver}

        return jsonify(result=out)

    @route('/logout/', methods=['GET', 'POST'], endpoint='logout')
    def logout(self):
        ''' logout from the system
        '''

        result = {'logout': 'success'}

        if 'loginready' in current_session:
            ready = current_session.pop('loginready')

        if 'name' in current_session:
            name = current_session.pop('name')

        request.environ['REMOTE_USER'] = None
        config.access = 'public'
        set_session_versions(config.release)
        setGlobalSession()
        logout_user()

        return redirect(url_for('index_page.Marvin:index'))

    @route('/login/', methods=['GET', 'POST'], endpoint='login')
    def login(self):
        form = processRequest(request=request)
        result = {}
        username = form['username'].strip()
        password = form['password'].strip()

        # do nothing if already validated
        if current_user.is_authenticated:
            return jsonify(result=result)

        # validate the user with htpassfile or trac username
        is_valid, user, result = validate_user(username, password, request=request)

        # get User only if valid
        if is_valid:
            user = get_db_user(username, password, dbsession=marvindb.session, user_model=marvindb.datadb.User, request=request)
            if user and user.check_password(password):
                login_user(user)

        if is_valid:
            result['status'] = 1
            result['message'] = 'Login Successful!'
            current_session['name'] = user.username
            current_session['loginready'] = True
            config.access = 'collab'
            setGlobalSession()
        else:
            result['status'] = -1
            result['message'] = 'Login {0} is not valid!'.format(username)

        return jsonify(result=result)


Marvin.register(index)
