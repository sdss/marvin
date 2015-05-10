from collections import OrderedDict
from json import loads

class Feedback:

    def __init__(self, session=None, username=None, auth=None):
        self.session = session
        self.set_ready()
        if not self.ready: self.set_member(username=username,auth=auth)
        self.set_counter(None)
        self.set_version()
        self.set_cube()
        self.set_category()
        self.message = ''
        self.status = -1
        self.comments = None
    
    def set_ready(self):
        if self.session is not None:
            if 'member_id' in self.session:
                try: self.ready = int(self.session['member_id']) > 0
                except: self.ready = False
            else: self.ready = False
        else: self.ready = False

    def set_session_member(self,id=None,username=None,auth=None):
        if self.session is not None:
            if 'member_id' not in self.session or self.session['member_id']!=id:
                try:
                    self.session['member_id'] = int(id)
                    self.session['member_username'] = username if username else 'None'
                    self.session['member_auth'] = auth if auth else 'None'
                    self.set_ready()
                except:
                    self.session['member_id'] = None
                    self.session['member_username'] = None
                    self.session['member_auth'] = None

    def set_member_from_session(self):
        if self.session is not None:
            member_id = self.session['member_id'] if 'member_id' in self.session else None
            member_username = self.session['member_username'] if 'member_username' in self.session else None
            member_auth = self.session['member_auth'] if 'member_auth' in self.session else None
            self.set_member(id=member_id,username=member_username,auth=member_auth,update_session=False)
        else: self.member = None

    def set_member(self,id=None,username=None,auth=None,update_session=True):
        if id is None: id = 1 if username is 'sdss' and auth is 'sdss' else None
        self.member = {'id':id,'username':username,'auth':auth}
        if update_session: self.set_session_member(id=id,username=username,auth=auth)
    
    def set_session_version(self,id=None,drp2ver=None,drp3ver=None,dapver=None):
        if self.session is not None:
            if 'version_id' not in self.session or self.session['version_id']!=id:
                try:
                    self.session['version_id'] = int(id)
                    self.session['version_drp2ver'] = drp2ver if drp2ver else 'None'
                    self.session['version_drp3ver'] = drp3ver if drp3ver else 'None'
                    self.session['version_dapver'] = dapver if dapver else 'None'
                except:
                    self.session['version_id'] = None
                    self.session['version_drp2ver'] = None
                    self.session['version_drp3ver'] = None
                    self.session['version_dapver'] = None

    def set_version_from_session(self):
        if self.session is not None:
            version_id = self.session['version_id'] if 'version_id' in self.session else None
            version_drp2ver = self.session['version_drp2ver'] if 'version_drp2ver' in self.session else None
            version_drp3ver = self.session['version_drp3ver'] if 'version_drp3ver' in self.session else None
            version_dapver = self.session['version_dapver'] if 'version_dapver' in self.session else None
            self.set_version(id=version_id,drp2ver=version_drp2ver,drp3ver=version_drp3ver,dapver=version_dapver,update_session=False)
        else: self.version = None

    def set_version(self,id=None,version=None,drp2ver=None,drp3ver=None,dapver=None,add=True,update_session=True):
        self.version = {'id':id,'drp2ver':drp2ver,'drp3ver':drp3ver,'dapver':dapver}
        if update_session: self.set_session_version(id=id,drp2ver=drp2ver,drp3ver=drp3ver,dapver=dapver)

    def set_counter(self,counter=None):
        if self.session is not None:
            if counter is not None:
                try: self.session['feedback_counter'] = int(counter)
                except: self.session['feedback_counter'] = None
            else: self.session['feedback_counter'] = self.session['feedback_counter'] + 1 if 'feedback_counter' in self.session else 1
    
    def set_session_category(self):
        if self.session is not None: self.session['feedback_category'] = self.category

    def set_category(self,update_session=True):
        if self.session is None or 'feedback_category' not in self.session:
            self.category = []
            for category in self.get_category():
                issues = [(issue['id'],issue['issue']) for issue in category['issues']]
                self.category.append((category['id'],OrderedDict([('category',category['category']),('key',category['key']),('issues',OrderedDict(issues))])))
            self.category = OrderedDict(self.category)
            if update_session: self.set_session_category()
        else:
            self.category = self.session['feedback_category']
            self.set_counter()

    def get_category(self):
        category = []
        categories = OrderedDict({1:'General',2:'Meta-data problems',3:'Target info',4:'Data problems'})
        keys = OrderedDict({1:'general',2:'metadata',3:'target',4:'data'})
        issues = OrderedDict({1:'wrong redshift',2:'wrong effective radius',3:'wrong inclination',4:'wrong center',5:'galaxy pair (physical)',6:'galaxy pair (projected)',7:'bright foreground star',8:'type I AGN (broad lines)',9:'gas/star kinematics misaligned',10:'poor sky subtraction',11:'flux calibration problems',12:'excess light at > 9000 A'})
        issue_id_per_category = [[],[1,2,3,4],[5,6,7,8,9],[10,11,12]]
        for i,issue_id in enumerate(issue_id_per_category): category.append({'id':i+1,'key':keys[i+1],'category':categories[i+1],'issues':[{'id':id,'issue':issues[id]} for id in issue_id]})
        return category


    def set_cube(self,plateid=None,ifuname=None,cube_pk=None):
        self.plateid = plateid
        self.ifuname = ifuname
        self.cube_pk = cube_pk
    
    def submit_comments(self,comments=[],issueids=[]):
        if [comment for comment in comments if comment] or issueids:
            self.message = "Please use the production site!"
            self.status = 1
        else:
            self.message = "Please enter either a comment or an issue before submitting feedback."
            self.status = 0

    def set_comments(self): pass
    
    def result(self):
        result = {'message':self.message,'status':self.status}
        if self.comments: result.extend({'comments':self.comments})
        if self.session and 'member_fullname' in self.session: result.extend({'membername':self.session['member_fullname']})
        return result
        
        



