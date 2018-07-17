# !usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-01-22 20:17:33
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2018-07-16 11:52:02

from __future__ import print_function, division, absolute_import
import requests
import os
try:
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urljoin
try:
    import matplotlib.pyplot as plt
    import mpl_toolkits.axes_grid1
    pyplot = True
except ImportError:
    pyplot = False
try:
    import ipwhois
except ImportError as e:
    ipwhois = None


class Sentry(object):
    ''' Interface to the Sentry API to grab and plot info

    Allows to query the Marvin2 Sentry page using their API.
    Can retrieve info on projects, events, issues, tags, etc.

    See doc here:
    https://docs.sentry.io/api/

    Parameters:
        org (str):
            Sentry organization name
        project (str):
            Sentry project name
        authtoken (str):
            token string for Sentry web api authentication

    '''

    def __init__(self, org='manga', project='marvin', authtoken=None):
        self.org = org
        self.project = project
        self.authtoken = authtoken
        self.data = None
        self.users = None
        self.tagvals = None
        self.ips = None
        self.locations = None
        self.response = None
        self.base_url = 'https://sentry.io/api/0/'
        self.header = {'Authorization': 'Bearer {authtoken}'}

    def check_auth(self):
        ''' checks and sets the authentication '''
        if self.authtoken:
            self.header['Authorization'] = self.header['Authorization'].format(**{'authtoken': self.authtoken})
        else:
            raise NameError('No authtoken set.  Please get one from the Sentry web api and set it!')

    def send_request(self, url, params=None):
        ''' Send the API request

        Parameters:
            url (str):
                The url to submit a request to
            params (list):
                list of optional parameters
        '''
        self.check_auth()
        self.response = requests.get(url, headers=self.header, params=params)
        self.check_response()

    def check_response(self):
        ''' checks the response

        Raises an error if the status is not 200. Otherwise sets gets the json data

        '''
        if not self.response.raise_for_status():
            self.data = self.response.json()

    def get_events(self):
        ''' Get the events for a projet '''
        url = urljoin(self.base_url, os.path.join('projects', self.org, self.project, 'events/'))
        self.send_request(url)
        self.events = self.data

    def get_users(self):
        ''' Get the users for a projet '''
        url = urljoin(self.base_url, os.path.join('projects', self.org, self.project, 'users/'))
        self.send_request(url)
        self.users = self.data

    def get_ips(self):
        ''' Get the ip addresses of a list of users '''
        if not self.users:
            self.get_users()
        self.ips = [u['ipAddress'] for u in self.users if u['ipAddress']]

    def lookup_ips(self, ip=None, method='whois'):
        ''' Look up the locations of the ips '''

        self.locations = []

        if ip and ip != '127.0.0.1':
            self.locations.append(self.get_ip_dict(ip, method=method))
        else:
            for ip in self.ips:
                if ip is not None and ip != '127.0.0.1':
                    self.locations.append(self.get_ip_dict(ip, method=method))

    def get_ip_dict(self, ip, method='whois'):
        ''' Get the ip lookup dictionary '''

        if not ipwhois:
            raise ImportError('Cannot look up ips.  You do not have the ipwhois package installed!')

        assert method in ['whois', 'rdap'], 'Method must either be rdap or whois'

        ipwho = ipwhois.IPWhois(ip)
        self.ipmethod = method
        ipdict = None
        if method == 'whois':
            try:
                ipdict = ipwho.lookup_whois()
            except Exception as e:
                print('Could not lookup ip {0}: {1}'.format(ip, e))
        elif method == 'rdap':
            try:
                ipdict = ipwho.lookup_rdap()
            except Exception as e:
                print('Could not lookup ip {0}: {1}'.format(ip, e))
        return ipdict

    def extract_locations(self):
        ''' Extraction the location info from the ip output '''
        self.places = []
        for loc in self.locations:
            nets = loc['nets'][0] if len(loc['nets']) > 0 else None
            locdict = {'asn_country_code': loc['asn_country_code'],
                       'place': {'city': nets['city'] if nets else None,
                                 'state': nets['state'] if nets else None,
                                 'country': nets['country'] if nets else None}}
            self.places.append(locdict)

    def get_project_tags(self):
        ''' Get the tags for a project '''
        url = urljoin(self.base_url, os.path.join('projects', self.org, self.project, 'tags/'))
        self.send_request(url)
        self.tags = self.data

    def list_tags(self):
        ''' List the tags available '''
        if self.tags:
            print([t['key'] for t in self.tags])
        else:
            print('No tags available')

    def get_tag_values(self, key=None):
        ''' Get the values for a given tag

        Parameters:
            key (str):
                The tag key to request the values for.  If not set, loops over all tags
        '''
        self.tagvals = {}

        if key is not None:
            self.tagvals[key] = self.get_value(key)
        else:
            for tag in self.tags:
                key = tag['key']
                self.tagvals[key] = self.get_value(key)

    def get_value(self, key):
        ''' Gets the values for single tag key

        Parameters:
            key (str):
                the tag key

        Returns:
            list of dictionaries of all values given the tag key

        '''
        url = os.path.join(self.base_url, 'projects', self.org, self.project, 'tags', key, 'values/')
        self.send_request(url)
        return self.data

    def get_place(self, place):
        ''' get a place from the places dictionary '''

        assert place in ['country', 'city', 'state'], 'Place must be either country, city, or state'
        return [p['place'][place] for p in self.places if p['place'][place]]

    def count_data(self, data):
        ''' Count the generic data and return a list of tuples '''
        return {(x, data.count(x)) for x in data}

    def get_pie_data(self, data):
        ''' Return the labels and sizes needed for the pie chart '''
        labels = [v[0] for v in data]
        sizes = [v[1] for v in data]
        return labels, sizes

    def plot_pie(self, key, data=None):
        ''' Makes a matplotlib pie chart for a given tag key '''
        if self.tagvals and key in self.tagvals:
            vals = self.tagvals[key]
            labels = [v['name'] for v in vals]
            sizes = [v['count'] for v in vals]
        elif key == 'place':
            assert data is not None, 'Must provide some data to extract'
            vals = self.count_data(data)
            labels, sizes = self.get_pie_data(vals)
        elif key == 'baseurl':
            vals = self.tagvals['url']
            names = [v['name'] for v in vals]
            labels = set(['sas' if 'sas.sdss.org' in n else 'api' for n in names])
            stats = [{'type': 'web' if l == 'sas' else l, 'count': sum(d['count'] for d in vals if l in d['name'])} for l in set(labels)]
            labels = [s['type'] for s in stats]
            sizes = [s['count'] for s in stats]
        else:
            sizes = None
            raise NameError('No sizes or labels found.  Make sure the data is somewhere')

        plt.clf()
        plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()




