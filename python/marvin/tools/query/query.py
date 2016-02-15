from __future__ import print_function
from marvin.tools.core import MarvinToolsClass, MarvinError


__all__ = ['Query']

class Query(MarvinToolsClass):

    def __init__(self, *args, **kwargs):

        super(Query, self).__init__(*args, **kwargs)

        # handle different modes
