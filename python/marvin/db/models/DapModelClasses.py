#!/usr/bin/env python
# encoding: utf-8
#
'''
new mangadap db model classes - Sept 7 , for alternate schema

'''

from __future__ import division, print_function

import re
from marvin.db.database import db
from marvin.utils.datamodel.dap import datamodel
import marvin.db.models.DataModelClasses as datadb
import numpy as np
from astropy.io import fits
from marvin.core.caching_query import RelationshipCache
from marvin.db.database import db
from marvin.utils.datamodel.dap import datamodel
from sqlalchemy import Float, ForeignKeyConstraint, and_, case, cast, select
from sqlalchemy.engine import reflection
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import configure_mappers, relationship
from sqlalchemy.schema import Column
from sqlalchemy.types import Integer


# ========================
# Define database classes
# ========================
Base = db.Base


def cameliseClassname(tableName):
    """Produce a camelised class name."""

    return str(tableName[0].upper() +
               re.sub(r'_([a-z])',
               lambda m: m.group(1).upper(), tableName[1:]))


def ClassFactory(name, tableName, BaseClass=db.Base, fks=None):
    tableArgs = [{'autoload': True, 'schema': 'mangadapdb'}]
    if fks:
        for fk in fks:
            tableArgs.insert(0, ForeignKeyConstraint([fk[0]], [fk[1]]))

    newclass = type(
        name, (BaseClass,),
        {'__tablename__': tableName,
         '__table_args__': tuple(tableArgs)})

    return newclass


class File(Base):
    __tablename__ = 'file'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<File (pk={0},name={1},tag={2})'.format(self.pk, self.filename, self.pipelineinfo.version.version)

    @property
    def is_map(self):
        return self.filetype.value == 'MAPS'

    @property
    def ftype(self):
        return self.filetype.value

    @property
    def partner(self):
        session = db.Session.object_session(self)
        return session.query(File).join(Structure, datadb.Cube, FileType).filter(
            Structure.pk == self.structure.pk, datadb.Cube.pk == self.cube.pk, FileType.pk != self.filetype.pk).one()

    @property
    def primary_header(self):
        primaryhdu = [h for h in self.hdus if h.extname.name == 'PRIMARY'][0]
        return primaryhdu.header

    @property
    def flux_header(self):
        ftype = self.filetype.value
        name = 'FLUX' if ftype == 'LOGCUBE' else 'EMLINE_GFLUX'
        fluxhdu = [h for h in self.hdus if h.extname.name == name][0]
        return fluxhdu.header

    @hybrid_property
    def quality(self):
        hdr = self.primary_header
        bits = hdr.get('DAPQUAL', None)
        if bits:
            return int(bits)
        else:
            return None

    @quality.expression
    def quality(cls):
        return select([HeaderValue.value.cast(Integer)]).\
                      where(and_(HeaderKeyword.pk==HeaderValue.header_keyword_pk,
                                 HduToHeaderValue.header_value_pk==HeaderValue.pk,
                                 HduToHeaderValue.hdu_pk==Hdu.pk,
                                 Hdu.file_pk==cls.pk,
                                 HeaderKeyword.name.ilike('DAPQUAL')
                        )).\
                      label('dapqual')

class CurrentDefault(Base):
    __tablename__ = 'current_default'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<CurrentDefault (pk={0},name={1})'.format(self.pk, self.filename)


class FileType(Base):
    __tablename__ = 'filetype'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<FileType (pk={0},value={1})'.format(self.pk, self.value)


class Hdu(Base):
    __tablename__ = 'hdu'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<Hdu (pk={0})'.format(self.pk)

    @property
    def header(self):
        '''Returns an astropy header'''

        session = db.Session.object_session(self)
        data = session.query(HeaderKeyword.name, HeaderValue.value,
                             HeaderValue.comment).join(HeaderValue, HduToHeaderValue).filter(
            HduToHeaderValue.header_value_pk == HeaderValue.pk,
            HduToHeaderValue.hdu_pk == self.pk).all()

        hdr = fits.Header(data)
        return hdr

    def header_to_dict(self):
        '''Returns a simple python dictionary header'''

        vals = self.header_values
        hdrdict = {str(val.keyword.name): val.value for val in vals}
        return hdrdict

    @property
    def name(self):
        return self.extname.name


class ExtName(Base):
    __tablename__ = 'extname'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<ExtName (pk={0}, name={1})'.format(self.pk, self.name)


class ExtType(Base):
    __tablename__ = 'exttype'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<ExtType (pk={0}, name={1})'.format(self.pk, self.name)


class ExtCol(Base):
    __tablename__ = 'extcol'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<ExtCol (pk={0}, name={1})'.format(self.pk, self.name)


class HduToExtCol(Base):
    __tablename__ = 'hdu_to_extcol'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<HduToExtcol (pk={0})'.format(self.pk)


class HduToHeaderValue(Base):
    __tablename__ = 'hdu_to_header_value'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<HduToHeaderValue (pk={0})'.format(self.pk)


class HeaderValue(Base):
    __tablename__ = 'header_value'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<HeaderValue (pk={0})'.format(self.pk)


class HeaderKeyword(Base):
    __tablename__ = 'header_keyword'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<HeaderKeyword (pk={0})'.format(self.pk)


class Structure(Base):
    __tablename__ = 'structure'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<Structure (pk={0}, bintype={1}, template={2})'.format(self.pk, self.bintype.name, self.template_kin.name)


class BinId(Base):
    __tablename__ = 'binid'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<BinId (pk={0}, id={1})'.format(self.pk, self.id)


def HybridRatio(line1, line2):
    ''' produces emission line ratio hybrid properties '''

    @hybrid_property
    def hybridRatio(self):

        if type(line1) == tuple:
            myline1 = getattr(self, line1[0]) + getattr(self, line1[1])
        else:
            myline1 = getattr(self, line1)

        if getattr(self, line2) > 0:
            return myline1 / getattr(self, line2)
        else:
            return -999.

    @hybridRatio.expression
    def hybridRatio(cls):

        if type(line1) == tuple:
            myline1 = getattr(cls, line1[0]) + getattr(cls, line1[1])
        else:
            myline1 = getattr(cls, line1)

        return cast(case([(getattr(cls, line2) > 0., myline1 / getattr(cls, line2)),
                          (getattr(cls, line2) == 0., -999.)]), Float)

    return hybridRatio


class SpaxelAtts(object):
    ''' New class to add attributes to all SpaxelProp classes '''
    pass

setattr(SpaxelAtts, 'ha_to_hb', HybridRatio('emline_gflux_ha_6564', 'emline_gflux_hb_4862'))
setattr(SpaxelAtts, 'nii_to_ha', HybridRatio('emline_gflux_nii_6585', 'emline_gflux_ha_6564'))
setattr(SpaxelAtts, 'oiii_to_hb', HybridRatio('emline_gflux_oiii_5008', 'emline_gflux_hb_4862'))
setattr(SpaxelAtts, 'oi_to_ha', HybridRatio('emline_gflux_oi_6302', 'emline_gflux_ha_6564'))
setattr(SpaxelAtts, 'sii_to_ha', HybridRatio(('emline_gflux_sii_6718', 'emline_gflux_sii_6732'), 'emline_gflux_ha_6564'))


def spaxel_factory(classname, clean=None):
    ''' class factory for the spaxelprop tables '''

    if clean:
        classname = 'Clean{0}'.format(classname)
    tablename = classname.lower()

    params = {'__tablename__': tablename, '__table_args__': {'autoload': True, 'schema': 'mangadapdb'}}
    if clean:
        params.update({'pk': Column(Integer, primary_key=True)})

    def newrepr(self):
        return '<{2} (pk={0}, file={1})'.format(self.pk, self.file_pk, classname)

    try:
        newclass = type(classname, (Base, SpaxelAtts,), params)
        newclass.__repr__ = newrepr
    except Exception as e:
        newclass = None

    return newclass


# create the (clean)spaxel models from the DAP datamodel
for dm in datamodel:
    classname = datamodel[dm].property_table
    newclass = spaxel_factory(classname)
    cleanclass = spaxel_factory(classname, clean=True)
    if newclass:
        locals()[classname] = newclass
    if cleanclass:
        locals()[cleanclass.__name__] = cleanclass


class BinMode(Base):
    __tablename__ = 'binmode'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<BinMode (pk={0}, name={1})'.format(self.pk, self.name)


class BinType(Base):
    __tablename__ = 'bintype'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<BinType (pk={0}, name={1})'.format(self.pk, self.name)


class ExecutionPlan(Base):
    __tablename__ = 'executionplan'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<ExecutionPlan (pk={0}, id={1})'.format(self.pk, self.id)


class Template(Base):
    __tablename__ = 'template'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<Template (pk={0}, name={1}, id={2})'.format(self.pk, self.name, self.id)


class ModelCube(Base):
    __tablename__ = 'modelcube'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<ModelCube (pk={0}, file={1})'.format(self.pk, self.file_pk)

    def get3DCube(self, extension='flux'):
        """Returns a 3D array of ``extension`` from the modelcube spaxels.

        For example, ``modelcube.get3DCube('flux')`` will return the original
        flux cube with the same ordering as the FITS data cube.

        Note that this method seems to be really slow retrieving arrays (this
        is especially serious for large IFUs).

        """

        session = db.Session.object_session(self)
        spaxels = session.query(getattr(ModelSpaxel, extension)).filter(
            ModelSpaxel.modelcube_pk == self.pk).order_by(ModelSpaxel.x, ModelSpaxel.y).all()

        # Assumes cubes are always square (!)
        nx = ny = int(np.sqrt(len(spaxels)))
        nwave = len(spaxels[0][0])

        spArray = np.array(spaxels)

        return spArray.transpose().reshape((nwave, ny, nx)).transpose(0, 2, 1)


class ModelSpaxel(Base):
    __tablename__ = 'modelspaxel'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<ModelSpaxel (pk={0}, mc={1})'.format(self.pk, self.modelcube_pk)


class RedCorr(Base):
    __tablename__ = 'redcorr'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<RedCorr (pk={0}, modelcube={1})'.format(self.pk, self.modelcube_pk)


class DapAll(Base):
    __tablename__ = 'dapall'
    __table_args__ = {'autoload': True, 'schema': 'mangadapdb'}

    def __repr__(self):
        return '<DapAll (pk={0}, file={1})'.format(self.pk, self.file_pk)


# -----
# Buld Relationships
# -----

File.cube = relationship(datadb.Cube, backref='dapfiles')
File.pipelineinfo = relationship(datadb.PipelineInfo, backref='dapfiles')
File.filetype = relationship(FileType, backref='files')
File.structure = relationship(Structure, backref='files')
CurrentDefault.file = relationship(File, uselist=False, backref='current_default')
Hdu.file = relationship(File, backref='hdus')
Hdu.extname = relationship(ExtName, backref='hdus')
Hdu.exttype = relationship(ExtType, backref='hdus')
Hdu.extcols = relationship(ExtCol, secondary=HduToExtCol.__table__, backref='hdus')
Hdu.header_values = relationship(HeaderValue, secondary=HduToHeaderValue.__table__, backref='hdus')
HeaderValue.keyword = relationship(HeaderKeyword, backref='values')
DapAll.file = relationship(File, uselist=False, backref='dapall')

Structure.executionplan = relationship(ExecutionPlan, backref='structures')
Structure.binmode = relationship(BinMode, backref='structures')
Structure.bintype = relationship(BinType, backref='structures')
Structure.template_kin = relationship(Template, foreign_keys=[Structure.template_kin_pk], backref='structures_kin')
Structure.template_pop = relationship(Template, foreign_keys=[Structure.template_pop_pk], backref='structures_pop')

insp = reflection.Inspector.from_engine(db.engine)

# add foreign key relationships on (Clean)SpaxelProp classses to File
spaxel_tables = {k: v for k, v in locals().items() if 'SpaxelProp' in k or 'CleanSpaxelProp' in k}

for classname, class_model in spaxel_tables.items():
    fks = insp.get_foreign_keys(class_model.__table__.name, schema='mangadapdb')
    if fks:
        backname = classname.lower().replace('prop', 'props')
        class_model.file = relationship(File, backref=backname)

fks = insp.get_foreign_keys(ModelCube.__table__.name, schema='mangadapdb')
if fks:
    ModelCube.file = relationship(File, backref='modelcube', uselist=False)

fks = insp.get_foreign_keys(ModelSpaxel.__table__.name, schema='mangadapdb')
if fks:
    ModelSpaxel.modelcube = relationship(ModelCube, backref='modelspaxels')

fks = insp.get_foreign_keys(RedCorr.__table__.name, schema='mangadapdb')
if fks:
    RedCorr.modelcube = relationship(ModelCube, backref='redcorr')

# ---------------------------------------------------------
# Test that all relationships/mappings are self-consistent.
# ---------------------------------------------------------

try:
    configure_mappers()
except RuntimeError as error:
    print('DapModelClasses, Error during configure_mapper: {0}'.format(error))

dap_cache = RelationshipCache(File.cube).\
               and_(RelationshipCache(File.filetype)).\
               and_(RelationshipCache(File.structure)).\
               and_(RelationshipCache(ModelCube.file)).\
               and_(RelationshipCache(ModelSpaxel.modelcube)).\
               and_(RelationshipCache(RedCorr.modelcube))
# add the SpaxelProp models to the cache
for classname, class_model in spaxel_tables.items():
    dap_cache = dap_cache.and_(RelationshipCache(class_model.file))

# delete extra classes from the various loops
del class_model
del cleanclass
del newclass
