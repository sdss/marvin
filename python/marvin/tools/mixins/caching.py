# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Filename: caching.py
# Project: mixins
# Author: Brian Cherinka
# Created: Monday, 27th January 2020 2:41:01 pm
# License: BSD 3-clause "New" or "Revised" License
# Copyright (c) 2020 Brian Cherinka
# Last Modified: Monday, 27th January 2020 6:35:50 pm
# Modified By: Brian Cherinka


from __future__ import print_function, division, absolute_import
from marvin import config
from marvin.db.caching import regions

__all__ = ['CacheMixIn']


cache_dict = {'Map': 'maps', 'Maps': 'maps', 'ModelCube': 'models'}


class CacheMixIn(object):
    ''' A mixin that provides optional caching of database tools
    
    When mixed in, determines the cache region to use based on the name
    of the class it's mixed with.  Can turn off caching altogether by setting
    use_db_tools_cache: False in custom marvin config yaml file.  If set,
    uses a null region cache.    
    '''
    _cache_regions = list(regions.keys())

    def __init__(self, region='default'):
        assert region in self._cache_regions, 'region must be a proper cache region'

        class_name = self.__class__.__name__

        use_tool_cache = config._custom_config.get('use_db_tools_cache', True)
        cache_region = 'null' if not use_tool_cache else cache_dict.get(
            class_name, region)
        self.cache_region = cache_region

