
from functools import wraps

# General Decorators
__all__ = ['parseRoutePath']


def parseRoutePath(f):
    ''' Decorator to parse generic route path '''
    @wraps(f)
    def decorated_function(inst, *args, **kwargs):
        if 'path' in kwargs and kwargs['path']:
            for kw in kwargs['path'].split('/'):
                if len(kw) == 0:
                    continue
                var, value = kw.split('=')
                kwargs[var] = value
        kwargs.pop('path')
        return f(inst, *args, **kwargs)
    return decorated_function
