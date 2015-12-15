
from __future__ import print_function
import requests,os, psutil

class API(object):
    """ This class defines convenience wrappers for the Marvin RESTful API """

    def __init__(self):
        self.results = None
        self.url = None
        self.params = None
        self.sasurl  = os.environ.get('SAS_URL',None)
        self.loggedin = None
        self._checkSASUrl()
        self.statuscodes = {200:'Ok',401:'Authentication Required',404:'URL Not Found',500:'Internal Server Error', 405:'Method Not Allowed', 400:'Bad Request'}
        if not self.loggedin: self._login()

    def _login(self):
        print('Need to login to sdss')
        self.loggedin = True

    def _checkSASUrl(self):
        if not self.sasurl:
            print('No SAS_URL environment variable set!')
        else:
            if 'localhost' in self.sasurl:
                print('Your SAS_URL is local..checking for running port')
                marvinlist = [proc.as_dict(attrs=['pid','name','cmdline']) for proc in psutil.process_iter() if 'python' in proc.name() and any(['run_marvin' in x for x in proc.cmdline()])]
                if any(marvinlist):
                    try: pindex = marvinlist[0]['cmdline'].index('-p')
                    except: pindex = None
                    if pindex: 
                        port = marvinlist[0]['cmdline'][pindex+1]
                        self.sasurl = 'http://localhost:{0}'.format(port)
                    else:
                        print('No port option found. Defaulting to SDSS SAS')
                        self.sasurl = 'https://sas.sdss.org'
                else:
                    print('No local running Marvins.  Defaulting to SDSS SAS')
                    self.sasurl = 'https://sas.sdss.org'

            sasprefix = os.environ.get('SAS_PREFIX',None)
            if not sasprefix:
                release = os.environ.get('MARVIN_RELEASE', 'mangawork')
                sasprefix = '' if 'localhost' in self.sasurl else 'marvin' if release == 'mangawork' else 'dr13/marvin'
            self.sasurl = os.path.join(self.sasurl,sasprefix)

    def _checkResponse(self,response):
        if response.status_code == 200:
            self.results = response.json()
        else:
            errmsg = 'Error accessing {0}: {1}'.format(response.url, self.statuscodes[response.status_code])
            self.results = {'http status code':response.status_code,'message':errmsg}

    def getMangaIdList(self):
        #self.url = 'http://localhost:5000/api/mangaids/'
        self.url = os.path.join(self.sasurl,'api/mangaids/')
        r = requests.get(self.url)
        self._checkResponse(r)
        return self.results

    def getMangaId(self, mangaid=None, version=None):
        #self.url = 'http://localhost:5000/api/mangaids/{0}/'.format(mangaid) if mangaid else None
        self.url = os.path.join(self.sasurl, 'api/mangaids/{0}'.format(mangaid)) if mangaid else None
        self.params = {'getver':version} if version else None

        if self.url:
            r = requests.get(self.url,params=self.params)
            self._checkResponse(r)

        return self.results

