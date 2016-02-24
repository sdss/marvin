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
drpclasses.pop('ArrayOps')


def formClassFactory(name, model, baseclass):
    ''' generate a new WTForm class based on SQLalchemy model class '''
    Meta = type('Meta', (object,), {'model': model})
    newclass = type(name, (baseclass,), {'Meta': Meta})
    return newclass

''' Build all the forms, currently this breaks due to WTForms-Alchemy not understanding ARRAY column types, can explicitly exclude those columns '''
# for key, val in drpclasses.items():
#     print(key, '-----')
#     classname = '{0}Form'.format(key)
#     newclass = formClassFactory(classname, val, ModelForm)
#     locals()[classname] = newclass

''' do we need an uber MarvinForm to combine all of these separate forms for easy reference? '''


class SampleForm(ModelForm):
    ''' test WTForm-Alchemy; WTForm based on SQLalchemy ModelClass '''
    class Meta:
        model = datadb.Sample

