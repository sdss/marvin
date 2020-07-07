# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Filename: caching.py
# Project: db
# Author: Brian Cherinka
# Created: Friday, 24th January 2020 2:47:54 pm
# License: BSD 3-clause "New" or "Revised" License
# Copyright (c) 2020 Brian Cherinka
# Last Modified: Monday, 27th January 2020 6:36:07 pm
# Modified By: Brian Cherinka


from __future__ import print_function, division, absolute_import
from marvin import config
from hashlib import md5
from dogpile.cache.region import make_region
import os
import copy

# DOGPILE CACHING SETUP

# dogpile cache regions.  A home base for cache configurations.
regions = {}

# create the dogpile directory for file-based caches
dogpath = os.environ.get('MANGA_SCRATCH_DIR', None)
if dogpath and os.path.isdir(dogpath):
    dogpath = dogpath
else:
    dogpath = os.path.expanduser('~')

dogroot = os.path.join(dogpath, 'dogpile_data')
if not os.path.isdir(dogroot):
    os.makedirs(dogroot)


# db hash key
def md5_key_mangler(key):
    """Receive cache keys as long concatenated strings;
    distill them into an md5 hash.

    """
    return md5(key.encode('ascii')).hexdigest()


def redis_key_mangler(key):
    ''' Set a redis key mangler
    
    Prefix the cache key for redis caches to distinguish dogpile
    caches in redis from other caches in redis.  See
    https://dogpilecache.sqlalchemy.org/en/latest/recipes.html#prefixing-all-keys-in-redis
    
    '''
    return "marvin:dogpile:" + key


# cache backend type
cache_type = {
    'null': {'name': 'dogpile.cache.null', 'args': {}},
    'file': {'name': 'dogpile.cache.dbm', 'args': {'filename': dogroot}},
    'redis': {'name': 'dogpile.cache.redis', 
              'args': {'url': os.environ.get('SESSION_REDIS'),
                       'distributed_lock': True}}
}


def make_new_region(name='cache.dbm', backend='file', expiration=3600, redis_exp_multiplier=2):
    ''' make a new dogpile cache region

    Creates a new dogpile cache region.  Default is to create a file-based
    cache called cache.dbm, with an expiration of 1 hour.

    Parameters:
        name (str):
            The name of the file-based cache
        backend (str):
            The type of cache backend to use.  Default is file.
        expiration (int):
            Expiration time in seconds of dogpile cache.  Default is 3600 seconds.
        redis_exp_multiplier (int):
            Integer factor to compute redis cache expiration time from dogpile expiration.
            Default is 2x the expiration.  
    '''
    # select the backend
    cache_backend = copy.deepcopy(cache_type[backend])

    # set the key mangler
    key_mangler = redis_key_mangler if backend == 'redis' else md5_key_mangler

    # create the cache filename
    if backend == 'file':
        fileroot = cache_backend['args'].get('filename', dogroot)
        cache_backend['args']['filename'] = os.path.join(fileroot, name)

    # update the redis expiration time
    if backend == 'redis':
        cache_backend['args']['redis_expiration_time'] = expiration * redis_exp_multiplier
        assert cache_backend['args']['url'] and cache_backend['args']['url'].startswith('redis'), \
            'Must have a redis url set to use a redis backend'

    # make the region
    reg = make_region(key_mangler=key_mangler).configure(
        cache_backend['name'],
        expiration_time=expiration,
        arguments=cache_backend['args']
    )
    return reg


# make a default cache region
regions['default'] = make_new_region()

# make a null cache for tests and dumps
regions['null'] = make_new_region(backend='null')

# make cache regions for NSA tables
for mpl in config._allowed_releases.keys():
    nsacache = 'nsa_{0}'.format(mpl.lower().replace('-', ''))
    regions[nsacache] = make_new_region(name=nsacache)

# make a maps redis cache
regions['maps'] = make_new_region(backend='redis')

# make a modelcube redis cache
regions['models'] = make_new_region(backend='redis')
