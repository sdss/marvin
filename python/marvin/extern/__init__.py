
import imp
import os
import sys
import warnings


def _import_module(name, relpath):
    """Imports a module from a path relative to this one's."""

    try:
        fp, pathname, description = imp.find_module(name)
        return imp.load_module(name, fp, pathname, description)
    except ImportError:
        pass

    try:
        path = os.path.join(os.path.dirname(__file__), relpath)
        fp, filename, description = imp.find_module(name, [path])
        return imp.load_module(name, fp, filename, description)
    except IOError:
        warnings.warn('Marvin cannot import {0}'.format(name), ImportWarning)
        return None


# Imports the external packages (some of them don't have __init__ in the right places and can
# not be imported normally).
# sdss_access = _import_module('sdss_access', 'sdss_access/python/')
sqlalchemy_boolean_search = _import_module(
    'sqlalchemy_boolean_search', 'sqlalchemy-boolean-search/')
wtforms_alchemy = _import_module('wtforms_alchemy', 'wtforms-alchemy/')
brain = _import_module('brain', 'marvin_brain/python')
