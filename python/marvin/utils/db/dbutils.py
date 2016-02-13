
from marvin import datadb, config
import traceback
import sys

# This line makes sure that "from marvin.utils.db.dbutils import *"
# will only import the functions in the list.
__all__ = ['get_traceback', 'testDbConnection']


def get_traceback(asstring=None):
    ''' Returns the traceback from an exception, a list

        Parameters:
        asstring = boolean to return traceback as a joined string
    '''
    ex_type, ex_info, tb = sys.exc_info()
    newtb = traceback.format_tb(tb)
    return ' '.join(newtb) if asstring else newtb


def testDbConnection(session=None):
    ''' Test the DB connection to '''

    res = {'good': None, 'error': None}

    #if not config.db:
    #    res['error'] = 'No database found!'
    #    return res

    if not session:
        from marvin import session

    try:
        tmp = session.query(datadb.PipelineVersion).first()
        res['good'] = True
    except Exception as e:
        error1 = 'Error connecting to manga database: {0}'.format(str(e))
        tb = get_traceback(asstring=True)
        error2 = 'Full traceback: {0}'.format(tb)
        error = ' '.join([error1, error2])
        res['error'] = error

    return res
