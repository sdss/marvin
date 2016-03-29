
import marvin
import traceback
import sys
import inspect

# This line makes sure that "from marvin.utils.db.dbutils import *"
# will only import the functions in the list.
__all__ = ['get_traceback', 'testDbConnection', 'generateClassDict']


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

    if not session:
        session = marvin.marvindb.session

    try:
        tmp = session.query(marvin.marvindb.datadb.PipelineVersion).first()
        res['good'] = True
    except Exception as e:
        error1 = 'Error connecting to manga database: {0}'.format(str(e))
        tb = get_traceback(asstring=True)
        error2 = 'Full traceback: {0}'.format(tb)
        error = ' '.join([error1, error2])
        res['error'] = error

    return res


def generateClassDict(modelclasses, lower=None):
    ''' Generates a dictionary of the Model Classes, based on class name as key, to the object class.
        Selects only those classes in the module with attribute __tablename__
        lower = True makes class name key all lowercase
    '''

    classdict = {}
    for model in inspect.getmembers(modelclasses, inspect.isclass):
        keyname = model[0].lower() if lower else model[0]
        if hasattr(model[1], '__tablename__'):
            classdict[keyname] = model[1]
    return classdict
