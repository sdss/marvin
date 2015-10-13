from collections import OrderedDict
from astropy.table import Table

class Inspection:

    cols = keys = ['membername','category','comment','issues','modified']
    trac_url = None

    def __init__(self, session=None, username=None, auth=None):
        self.session = session
        self.message = ''
        self.status = -1
        self.set_ready()
        if self.ready: self.set_member(id=self.session['member_id'],update_session=False)
        if not self.ready: self.set_member(username=username,auth=auth)
        self.set_super()
        self.ready=True
        self.set_counter(None)
        self.set_drpver()
        self.set_dapver()
        self.set_ifudesign()
        self.set_cube()
        self.set_category(forcomment=True)
        self.set_category(fordapqa=True)
        self.options = None
        self.panels = None
        self.comment = None
        self.option = None
        self.panel = None
        self.comments = None
        self.dapqacomments = None
        self.dapqasearchcomments = None
        self.dapqatags = None
        self.totaldapcomments = None
        self.cubecomments = None
        self.dapqacubecomments = None
        self.tags = {'1':'test','2':'daptest'}
        self.alltags = {'1':'test','2':'elliptical','3':'galaxy','4':'daptest','5':'hello','6':'world','7':'new','8':'hope'}
        self.cubetags = {'12701':['hello','world'], '1901':['new'],'9101':['daptest','hello','new','test','challenge','hope']}
        self.searchcomments = None
        self.recentcomments = {'1':['Hello'],'2':[],'3':[],'4':[],'5':[]}
        self.ticket = None
        self.feedback = None
        self.feedbacks = None
        self.feedbacksubject = None
        self.feedbackfield = None
        self.feedbacktype =  None
        self.feedbackproduct =  None
        self.feedbackstatuses =  ['Submitted', 'Opened', 'Resolved' ,'Will Not Fix']
        self.feedbackstatus = None
        self.feedbackvote = None
        self.tracticket =  None
        self.tracticketid =  None
        self.cols = self.keys = []
        self.dapqacols = self.dapqakeys = []
        self.category_id = None
        self.issueids = None
        self.memberids = None
        self.keyword = None
        self.date = None
        self.component = {}
        self.type = {}
        self.set_dapqaoptions()
    
    def set_super(self):
        self.super = True

    def set_session_dapqaoptions(self):
        if self.session is not None: self.session['dapqaoptions'] = self.dapqaoptions

    def set_dapqaoptions(self):
        if 'dapqaoptions' in self.session: self.dapqaoptions = self.session['dapqaoptions']
        else:
            dapqacats = self.get_category(fordapqa=True)
            self.dapqaoptions = {}
            self.dapqaoptions.update({'defaulttitle':{dapqacat['key']:dapqacat['category'] for dapqacat in dapqacats}})
            self.dapqaoptions.update({'maptype':{'kin':'Kinematic','snr':'SNR','binnum':'Bin_Num','emflux':'EMflux','emfluxew':'EMflux_EW','emfluxfb':'EMflux_FB'}})
            self.dapqaoptions.update({'subfolder':{'maps':'maps','spectra':'spectra','radgrad':'gradients'}})
            self.dapqaoptions.update({'bindict':{'none':'NONE','all':'ALL','ston':'STON','rad':'RADIAL'}})
            self.set_session_dapqaoptions()
    
    def get_panelnames(self,mapid,bin=None):
        if 'emflux' in mapid:
            panelname = [(0,'oii'),(1,'hbeta'),(2,'oiii'),(3,'halpha'),(4,'nii'),(5,'sii')]
        elif 'snr' in mapid:
            panelname = [(0,'signal'),(1,'noise'),(2,'snr'),(3,'halpha_ew'),(4,'resid'),(5,'chisq')]
        elif 'kin' in mapid:
            if 'ston' in bin: panelname = [(0,'emvel'),(1,'emvdisp'),(2,'sth3'),(3,'stvel'),(4,'stvdisp'),(5,'sth4')]
            elif 'none' in bin: panelname = [(0,'emvel'),(1,'emvdisp'),(2,'chisq'),(3,'stvel'),(4,'stvdisp'),(5,'resid')]
        elif 'binnum' in mapid:
            panelname = [(0,'spaxel number')]
        elif 'specmap' in mapid:
            panelname = [(0,'oii'),(1,'hbeta'),(2,'oiii'),(3,'halpha'),(4,'nii'),(5,'sii')]
        else: panelname = [(None,'spectrum')]
        return panelname 
        
    def get_dapmapdict(self,key,bin=None):
        kinlist = ['vel_map','vdisp_map','sth'] if 'ston' in bin else ['vel_map','vdisp_map','chisq','resid']
        if key == 'maps':
            mapdict = {'emfluxew':'ew_map','emfluxfb':'fb_map','kin': kinlist, 'snr':['noise', 'signal', 'snr', 'halpha_ew','chisq','resid'],'binnum':'bin_num'}
        elif key == 'radgrad':
            mapdict = {'emflux':'gradient'}
        elif key == 'spectra':
            mapdict = {'spec1':'spec'}
        return mapdict

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
        self.set_session_drpver(id=id,drp2ver=drp2ver,drp3ver=drp3ver)
        self.set_session_dapver(dapver=dapver)

    def set_session_drpver(self,id=None,drp2ver=None,drp3ver=None):
        if self.session is not None:
            if 'drpver_id' not in self.session or self.session['drpver_id']!=id:
                try:
                    self.session['drpver_id'] = int(id)
                    self.session['drpver_drp2ver'] = drp2ver if drp2ver else 'None'
                    self.session['drpver_drp3ver'] = drp3ver if drp3ver else 'None'
                except:
                    self.session['drpver_id'] = None
                    self.session['drpver_drp2ver'] = None
                    self.session['drpver_drp3ver'] = None

    def set_session_dapver(self,id=None,dapver=None):
        if self.session is not None:
            if 'dapver_id' not in self.session or self.session['dapver_id']!=id:
                try:
                    self.session['dapver_id'] = int(id)
                    self.session['dapver_dapver'] = dapver if dapver else 'None'
                except:
                    self.session['dapver_id'] = None
                    self.session['dapver_dapver'] = None

    def set_version_from_session(self):
        self.set_drpver_from_session()
        self.set_dapver_from_session()

    def set_drpver_from_session(self):
        if self.session is not None:
            drpver_id = self.session['drpver_id'] if 'drpver_id' in self.session else None
            drpver_drp2ver = self.session['drpver_drp2ver'] if 'drpver_drp2ver' in self.session else None
            drpver_drp3ver = self.session['drpver_drp3ver'] if 'drpver_drp3ver' in self.session else None
            self.set_drpver(id=drpver_id,drp2ver=drpver_drp2ver,drp3ver=drpver_drp3ver,update_session=False)
        else: self.drpver = None

    def set_dapver_from_session(self):
        if self.session is not None:
            dapver_id = self.session['dapver_id'] if 'dapver_id' in self.session else None
            dapver_dapver = self.session['dapver_dapver'] if 'dapver_dapver' in self.session else None
            self.set_dapver(id=dapver_id,dapver=dapver_dapver,update_session=False)
        else: self.dapver = None

    def set_version(self,id=None,drpver=None,drp2ver=None,drp3ver=None,dapver=None,add=True,update_session=True):
        self.set_drpver(drpver=drpver,drp2ver=drp2ver,drp3ver=drp3ver,add=add,update_session=True)
        self.set_dapver(dapver=dapver,add=add,update_session=True)

    def set_drpver(self,id=None,drpver=None,drp2ver=None,drp3ver=None,add=True,update_session=True):
        self.drpver = {'id':id,'drp2ver':drp2ver,'drp3ver':drp3ver}
        if update_session: self.set_session_drpver(id=id,drp2ver=drp2ver,drp3ver=drp3ver)

    def set_dapver(self,id=None,dapver=None,add=True,update_session=True):
        self.dapver = {'id':id,'dapver':dapver}
        if update_session: self.set_session_dapver(id=id,dapver=dapver)

    def set_options(self):
        self.options = {'bintype': [u'all5', u'none2', u'rad1', u'rad2', u'rad3', u'rad4', u'ston1'],
        'maptype': [u'emflux',  u'emfluxew',  u'emfluxfb',  u'kin',  u'snr',  u'spec0',  u'spec1',  u'spec8'],
        'mode': [u'cube', u'rss']}
    
    def set_panels(self):
        self.panels = [u'chisq',
         u'emvdisp',
         u'emvel',
         u'halpha',
         u'halpha_ew',
         u'hbeta',
         u'nii',
         u'noise',
         u'oii',
         u'oiii',
         u'resid',
         u'signal',
         u'sii',
         u'snr',
         u'spectrum',
         u'sth3',
         u'sth4',
         u'stvdisp',
         u'stvel']

    def set_option(self,id=None,mode=None,bintype=None,maptype=None,specpanel=None,add=True):
        self.option = {'id':id,'mode':mode,'bintype':bintype,'maptype':maptype,'specpanel':specpanel}

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
    def retrieve_dapqasearchcomments(self):
        self.cols = self.keys = ['membername','plate','ifuname','drpver','category','comment','issues','modified']

    def retrieve_dapqacubecomments(self):
        self.dapqacols = self.dapqakeys = ['membername','category','mode','bintype','maptype','panel','comment','issues','modified']
        self.dapqacubecomments = {'9101':Table([{'membername':'Brian','category':'maps','mode':'cube','bintype':'none2','maptype':'kin',
        'panel':'oii','comment':'this is great','issues':['1','2','3'],'modified':'today'}])}

    def retrieve_cubetags(self): pass
    def retrieve_searchcomments(self):
        self.cols = self.keys = ['membername','plate','ifuname','drpver','category','comment','issues','modified']
    
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
    
    def drop_dapqatags_from_session(self): pass
    def drop_dapqacomments_from_session(self): pass

    def set_dapqacomments(self): 
        self.status=1
        self.message=''

    def retrieve_dapqacomments(self,catid=None): 
        self.message = "Please use the production site!"
        self.status = 1
        self.dapqacomments = []
        self.dapqacomments.append({'catid':'1','position':'1','panel':'OII','issues':['1','4','5'],'comment':'Here is a comment'})
        self.set_totaldapcomments()
        
    def set_totaldapcomments(self):
        ncomment = sum([1 for pancom in self.dapqacomments if pancom['comment']])
        nissues = sum([len(pancom['issues']) for pancom in self.dapqacomments])
        self.totaldapcomments = "You have entered {0} comments and {1} issues.".format(ncomment,nissues)

    def set_session_dapqacomments(self,catid=None,comments=None,touched=None):
        self.status = 1
        self.message = 'Failed to save the previous comments in the session!'
    def set_session_tags(self,tags=[]): pass
    def reset_dapqacomments(self): pass
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
        self.feedbacks = Table([{'id':1,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':2,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':3,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':4,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':5,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':6,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':7,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':8,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':9,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':10,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':11,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':12,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':13,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0},{'id':14,'membername':'Brian','subject':'test','feedback':'some feedback',
            'type':'Bug','product':'Marvin','tracticket':'promote','modified':'now','status':'Submitted','vote':0}])
        self.cols = self.keys = ['id','membername','subject','feedback','type','product','tracticket','modified','status','vote']
    
    def set_feedback(self,id=None): self.feedback = None
    def update_feedback(self,status=None): self.feedbackstatus = status
    def vote_feedback(self,vote=None): 
        self.feedbackvote = vote
        self.status = 1
         
    def promote_tracticket(self): pass

    def result(self):
        result = {'ready':self.ready,'message':self.message,'status':self.status,'alltags':self.alltags}
        if self.session and 'member_fullname' in self.session: result.update({'membername':self.session['member_fullname']})
        if self.comments: result.update({'comments':self.comments})
        if self.recentcomments: result.update({'recentcomments':self.recentcomments})
        if self.dapqacomments: result.update({'dapqacomments':self.dapqacomments})
        if self.tags: result.update({'tags':self.tags})
        if self.totaldapcomments: result.update({'totaldapcomments':self.totaldapcomments})
        return result
