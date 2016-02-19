from __future__ import print_function
from wtforms import Form, StringField, validators
from marvin.tools.core import MarvinToolsClass, MarvinError


__all__ = ['Query']


class Query(MarvinToolsClass):

    def __init__(self, *args, **kwargs):

        super(Query, self).__init__(*args, **kwargs)

        # handle different modes

    def set_params():
        """Set parameters."""
        pass

    def add_condition():
        """Add a condition."""
        # if self.mode == 'remote':
        # elif self.mode == 'api':
        pass

    def run():
        """Get data from API."""
        pass
