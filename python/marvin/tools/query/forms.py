#!/usr/bin/env python
# encoding: utf-8

'''
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-02-23 14:13:28 by Brian Cherinka
    Last Modified On: 2016-02-23 14:13:28 by Brian

'''
from __future__ import print_function
from __future__ import division
from marvin import session, datadb, config
from marvin.utils.db import generateClassDict

# flask_wtf does not work locally - OUTSIDE APPLICATON CONTEXT; need some kind of toggle for web version and not???
if config._inapp:
    from flask_wtf import Form
else:
    from wtforms import Form

from wtforms import StringField, validators, SelectField
from wtforms.widgets import Select
from wtforms_alchemy import model_form_factory

__all__ = ['TestForm', 'SampleForm']

# Base form class
BaseModelForm = model_form_factory(Form)


class ModelForm(BaseModelForm):
    ''' sub class a new ModelForm so it works with Flask-WTF in APP mode;  for auto CSRF tokens...who knows...'''
    @classmethod
    def get_session(self):
        return session


class TestForm(Form):
    ''' test WTF-Form ; allows for manip. of validation, custom rendering, widget, etc '''
    redshift = StringField('NSA Redshift', [validators.Length(min=4, max=25)])
    _ifus = sorted(list(set([i.name[:-2] for i in session.query(datadb.IFUDesign).all()])), key=lambda t: int(t))
    _ifufields = [('ifu{0}'.format(_i), _i) for _i in _ifus]
    ifu = SelectField('IFU Design', choices=_ifufields)


''' builds a dict. for modelclasses with key ClassName and value SQLalchemy model class ; '''
drpclasses = generateClassDict(datadb, filterby='DataModelClasses')
out = ['ArrayOps', 'Plate']
tmp = [drpclasses.pop(o) for o in out]
# import sdss.internal.database.utah.mangadb.SampleModelClasses as sampledb
# sampclasses = generateClassDict(sampledb, filterby='SampleModelClasses')
# out = ['Character']
# tmp = [sampclasses.pop(o) for o in out]
# import sdss.internal.database.utah.mangadb.DapModelClasses as dapdb
# dapclasses = generateClassDict(dapdb, filterby='DapModelClasses')


def formClassFactory(name, model, baseclass):
    ''' generate a new WTForm class based on SQLalchemy model class '''
    Meta = type('Meta', (object,), {'model': model})
    newclass = type(name, (baseclass,), {'Meta': Meta})
    return newclass

opdict = {'le': '<=', 'ge': '>=', 'gt': '>', 'lt': '<', 'ne': '!=', 'eq': '='}
ops = [(key, val) for key, val in opdict.items()]
# operator = SelectField(u'Operator', choices=ops)


class SampleForm(ModelForm):
    ''' test WTForm-Alchemy; WTForm based on SQLalchemy ModelClass '''
    class Meta:
        model = datadb.Sample
    operator = SelectField(u'Operator', choices=ops)


class MarvinForm(object):

    def __init__(self):
        self._param_form_lookup = {}
        self._generateFormClasses(drpclasses)
        # self._generateFormClasses(sampclasses)
        # self._generateFormClasses(dapclasses)

    def _generateFormClasses(self, classes):
        for key, val in classes.items():
            print(key, val, '----')
            classname = '{0}Form'.format(key)
            newclass = formClassFactory(classname, val, ModelForm)
            # newclass.operator = operator
            self.__setattr__(classname, newclass)
            self._loadParams(classname, newclass)

    def _loadParams(self, classname, newclass):
        for key in newclass.__dict__.keys():
            if key[:1] != '_' and 'Meta' not in key:
                self._param_form_lookup[key] = newclass

    def callInstance(self, form, params=None):
        ''' create an instance of a specified WTForm '''
        return form(**params) if params else form()


