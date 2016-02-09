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
        #self._checkSASUrl()
        self.statuscodes = {200:'Ok',401:'Authentication Required',404:'URL Not Found',500:'Internal Server Error', 405:'Method Not Allowed', 400:'Bad Request'}
        #if not self.loggedin: self._login()

    def _checkResponse(self,response):
        if response.status_code == 200:
            self.results = response.json()
        else:
            errmsg = 'Error accessing {0}: {1}'.format(response.url, self.statuscodes[response.status_code])
            self.results = {'http status code':response.status_code,'message':errmsg}
    
    def getCube(self,mangaid=None):
        self.url = 'http://5aafb8e.ngrok.com/api/cubes/{0}/'.format(mangaid) if mangaid else None
        #self.url = os.path.join(self.sasurl, 'api/mangaids/{0}'.format(mangaid)) if mangaid else None
        #self.params = {'getver':version} if version else None

        if self.url:
            r = requests.get(self.url,params=self.params)
            self._checkResponse(r)

        return self.results