from collections import OrderedDict

class Inspection:

    cols = keys = ['membername','category','comment','issues','modified']

    def __init__(self, session=None, username=None, auth=None):
        self.session = session
        self.message = ''
        self.status = -1
        self.set_ready()
        if self.ready: self.set_member(id=self.session['member_id'],update_session=False)
        if not self.ready: self.set_member(username=username,auth=auth)
        self.ready=True
        self.set_counter(None)
        self.set_version()
        self.set_ifudesign()
        self.set_cube()
        self.set_category(forcomment=True)
        self.set_category(fordapqa=True)
        self.comment = None
        self.option = None
        self.panel = None
        self.comments = None
        self.dapqacomments = None
        self.cubecomments = None
        self.tags = {'1':'test'}
        self.alltags = {'1':'test','2':'elliptical','3':'galaxy'}
        self.cubetags = {'12701':['hello','world'], '1901':['new','test','challenge','hope']}
        self.searchcomments = None
        self.recentcomments = {'1':['Hello'],'2':[],'3':[],'4':[],'5':[]}
        self.ticket = None
        self.feedback = None
        self.feedbacks = None
        self.feedbacksubject = None
        self.feedbackfield = None
        self.feedbacktype =  None
        self.feedbackproduct =  None
        self.tracticket =  None
        self.tracticketid =  None
        self.cols = self.keys = []
        self.category_id = None
        self.issueids = None
        self.memberids = None
        self.keyword = None
        self.date = None
        self.component = {}
        self.type = {}

    def set_component(self): self.component = OrderedDict([('Marvin','marvin'),('DRP','mangadrp'),('DAP','mangadap'),('Mavis','mangacas')])
    def set_type(self): self.type = OrderedDict([('Feature Request','enhancement'), ('Bug','defect'), ('Use Case','task'), ('Other','task')])

    def set_ready(self):
        if self.session is not None:
            if 'member_id' in self.session:
                try: self.ready = int(self.session['member_id']) > 0
                except: self.ready = False
            else: self.ready = False
        else: self.ready = False

    def set_session_member(self,id=None,username=None,fullname=None,auth=None):
        if self.session is not None:
            if 'member_id' not in self.session or self.session['member_id']!=id:
                try:
                    self.session['member_id'] = int(id)
                    self.session['member_username'] = username if username else 'None'
                    self.session['member_fullname'] = fullname if fullname else 'None'
                    self.session['member_auth'] = auth if auth else 'None'
                    self.set_ready()
                except:
                    self.session['member_id'] = None
                    self.session['member_username'] = None
                    self.session['member_fullname'] = None
                    self.session['member_auth'] = None

    def set_member_from_session(self):
        if self.session is not None:
            member_id = self.session['member_id'] if 'member_id' in self.session else None
            member_username = self.session['member_username'] if 'member_username' in self.session else None
            member_auth = self.session['member_auth'] if 'member_auth' in self.session else None
            self.set_member(id=member_id,username=member_username,auth=member_auth,update_session=False)
        else: self.member = None

    def set_member(self,id=None,username=None,auth=None,add=False,update_session=True):
        if id is None: id = 1 if (username,auth)==('sdss','43799f65a46144a0535ccea32fe2af34') else 0
        elif id==1: username,auth=('sdss','43799f65a46144a0535ccea32fe2af34')
        if username and auth:
            self.member = {'id':int(id),'username':username,'auth':auth}
            fullname = "SDSS User" if id==1 else "Uknown user"
            if self.member['id']:
                self.status = 1
                self.message = "Logged in as %s." % self.member['username']
            else:
                self.status = 0
                self.message = "Failed login for %s. " % username
                self.message += "Please retry." if username=='sdss' else "Username unrecognized."
        else: self.member = None
        if self.member and update_session: self.set_session_member(id=self.member['id'],username=self.member['username'],fullname=fullname,auth=self.member['auth'])

    
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

    def set_option(self,id=None,mode=None,bintype=None,maptype=None,add=True):
        self.option = {'id':id,'mode':mode,'bintype':bintype,'maptype':maptype}

    def set_panel(self,id=None,panel=None,position=None,add=True):
        self.panel = {'id':id,'panel':panel,'position':position}

    def set_counter(self,counter=None):
        if self.session is not None:
            if counter is not None:
                try: self.session['inspection_counter'] = int(counter)
                except: self.session['inspection_counter'] = None
            else: self.session['inspection_counter'] = self.session['inspection_counter'] + 1 if 'inspection_counter' in self.session else 1
    
    def set_session_category(self,forcomment=None,fordapqa=None):
        session_variable = 'inspection_category' if forcomment else 'inspection_dapqacategory' if fordapqa else None
        if self.session is not None and session_variable:
            self.session[session_variable] = self.category if forcomment else self.dapqacategory if fordapqa else None

    def set_category(self,forcomment=None,fordapqa=None,update_session=True):
        session_variable = 'inspection_category' if forcomment else 'inspection_dapqacategory' if fordapqa else None
        if self.session is None or session_variable not in self.session:
            clist = []
            for category in self.get_category(forcomment=forcomment,fordapqa=fordapqa):
                issues = [(issue['id'],issue['issue']) for issue in category['issues']]
                clist.append((category['id'],OrderedDict([('category',category['category']),('key',category['key']),('issues',OrderedDict(issues))])))
            if forcomment: self.category = OrderedDict(clist)
            elif fordapqa: self.dapqacategory = OrderedDict(clist)
            if update_session: self.set_session_category(forcomment=forcomment,fordapqa=fordapqa)
        else:
            if forcomment: self.category = self.session[session_variable]
            elif fordapqa: self.dapqacategory = self.session[session_variable]
            self.set_counter()

    def get_category(self,forcomment=None,fordapqa=None):
        category = []
        if forcomment:
            categories = {1:'General',2:'Meta-data problems',3:'Target info',4:'Data problems',5:'DAP QA'}
            keys = {1:'general',2:'metadata',3:'target',4:'data',5:'dapqa'}
            issues = {1:'wrong redshift',2:'wrong effective radius',3:'wrong inclination',4:'wrong center',5:'galaxy pair (physical)',
                 6:'galaxy pair (projected)',7:'bright foreground star',8:'type I AGN (broad lines)',9:'gas/star kinematics misaligned',
                 10:'poor sky subtraction',11:'flux calibration problems',12:'excess light at > 9000 A'}
            issue_id_per_category = [[],[1,2,3,4],[5,6,7,8,9],[10,11,12]]
            for i,issue_id in enumerate(issue_id_per_category): category.append({'id':i+1,'key':keys[i+1],'category':categories[i+1],'issues':[{'id':id,'issue':issues[id]} for id in issue_id]})
        elif fordapqa:
            categories = {1:'Maps',2:'Radial Gradients',3:'Spectra'}
            keys = {1:'maps',2:'radgrad',3:'spectra'}
            #subkeys = {'maps':['kin','snr','emfluxew','emfluxfb'],'radgrad':['emflux'],'spectra':[]}
            issues = {1:'Bad Color Scale',2:'High Chi^2 (w/ small residuals)',3:'Discontinuities',4:'Irregular Kinematics', 
                 5: 'Satellite(s)?',6:'Background/Foreground Galaxy?',7:'Foreground Star',
                 8:'Large Differences between Wang & Belfiore gradients',
                 9:'Poor Sky Subtraction (unmasked)',10:'Poor Continuum Fit',11:'Dichroic Dip',
                 12:'Strongly nonGaussian emission lines',
                 13:'Poor Wang OII', 14:'Poor Wang Hbeta', 15:'Poor Wang OIII', 16:'Poor Wang NII', 17:'Poor Wang Halpha', 18:'Poor Wang SII',
                 19:'Poor Belfiore OII', 20:'Poor Belfiore Hbeta', 21:'Poor Belfiore OIII', 22:'Poor Belfiore NII', 23:'Poor Belfiore Halpha', 
                 24:'Poor Belfiore SII'}
            issue_id_per_category = [[1,2,3,4,5,6,7],[8],[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
            for i,issue_id in enumerate(issue_id_per_category): category.append({'id':i+1,'key':keys[i+1],'category':categories[i+1],'issues':[{'id':id,'issue':issues[id]} for id in issue_id]})
        return category

    def set_ifudesign(self,plateid=None,ifuname=None):
        self.plateid = plateid
        self.ifuname = ifuname
        self.ifudesign = None

    def set_cube(self,cubepk=None):
        self.cubepk = cubepk
    
    def retrieve_comments(self): pass
    def retrieve_alltags(self,ids=False): pass
    def retrieve_tags(self): pass
    def retrieve_cubecomments(self): pass
    def retrieve_cubetags(self): pass
    def retrieve_searchcomments(self): pass
    def set_search_parameters(self,form=None): pass
    def refine_cubes_by_tagids(self, tagids=None, cubes=None): pass
    
    def submit_comments(self,comments=[],issueids=[],tags=[]):
        if [comment for comment in comments if comment] or issueids or tags:
            self.message = "Please use the production site!"
            self.status = 1
        else:
            self.message = "Please enter either a comment or an issue before submitting."
            self.status = 0
    
    def set_comments(self): pass
    
    def set_dapqacomments(self): pass

    def retrieve_dapqacomments(self,catid=None): pass
    def set_session_dapqacomments(self,catid=None,comments=None): pass
    
    def submit_dapqacomments(self): pass
    
    def set_recentcomments(self):
        self.recentcomments = {}
        for category_id,category in self.category.items(): self.recentcomments.update({category_id:[]})
    
    def submit_feedback(self,form={}):
        self.message = "Please use the production site!"
        self.status = 1

    def submit_tracticket(self):  pass

    def retrieve_feedbacks(self):
        self.set_feedbacks()
        self.status = 1
    
    def set_feedbacks(self):
        self.feedbacks = {}
    
    def set_feedback(self,id=None): self.feedback = None
    
    def promote_tracticket(self): pass

    def result(self):
        result = {'ready':self.ready,'message':self.message,'status':self.status,'alltags':self.alltags}
        if self.session and 'member_fullname' in self.session: result.update({'membername':self.session['member_fullname']})
        if self.comments: result.update({'comments':self.comments})
        if self.recentcomments: result.update({'recentcomments':self.recentcomments})
        if self.tags: result.update({'tags':self.tags})
        return result
