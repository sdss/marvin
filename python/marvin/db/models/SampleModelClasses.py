#!/usr/bin/env python
# encoding: utf-8
"""
ModelClasses.py

Created by José Sánchez-Gallego on 23 Jul 2015.
Licensed under a 3-clause BSD license.

Revision history:
    23 Jul 2015 J. Sánchez-Gallego
      Initial version
    21 Feb 2016 J. Sánchez-Gallego
      Rewritten as classes derived from declarative base.

"""

from __future__ import division, print_function

import itertools
import math
import re
import shutil

import numpy as np
from marvin.core.caching_query import RelationshipCache
from marvin.db.database import db
from sqlalchemy import Float, ForeignKeyConstraint, case, cast, func
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.inspection import inspect as sa_inspect
from sqlalchemy.orm import backref, configure_mappers, relationship


try:
    import cStringIO as StringIO
except ImportError:
    from io import StringIO


Base = db.Base


# The softening parameter for asinh magnitudes
bb = {'u': 1.4e-10,
      'g': 0.9e-10,
      'r': 1.2e-10,
      'i': 1.8e-10,
      'z': 7.4e-10}


def cameliseClassname(tableName):
    """Produce a camelised class name."""

    return str(tableName[0].upper() +
               re.sub(r'_([a-z])',
               lambda m: m.group(1).upper(), tableName[1:]))


def ClassFactory(name, tableName, BaseClass=db.Base, fks=None):
    tableArgs = [{'autoload': True, 'schema': 'mangasampledb'}]
    if fks:
        for fk in fks:
            tableArgs.insert(0, ForeignKeyConstraint([fk[0]], [fk[1]]))

    newclass = type(
        name, (BaseClass,),
        {'__tablename__': tableName,
         '__table_args__': tuple(tableArgs)})

    return newclass


class MangaTarget(Base):
    __tablename__ = 'manga_target'
    __table_args__ = {'autoload': True, 'schema': 'mangasampledb'}

    def __repr__(self):
        return '<MangaTarget (pk={0}, mangaid={1})>'.format(self.pk,
                                                            self.mangaid)


class Anime(Base):
    __tablename__ = 'anime'
    __table_args__ = {'autoload': True, 'schema': 'mangasampledb'}

    def __repr__(self):
        return '<Anime (pk={0}, anime={1})>'.format(self.pk, self.anime)


class Character(Base):
    __tablename__ = 'character'
    __table_args__ = {'autoload': True, 'schema': 'mangasampledb'}

    target = relationship(MangaTarget, backref='character', uselist=False)
    anime = relationship(Anime, backref='characters')

    def __repr__(self):
        return '<Character (pk={0}, name={1})>'.format(self.pk, self.name)

    def savePicture(self, path):
        """Saves the picture blob to disk."""

        buf = StringIO(self.picture)
        with open(path, 'w') as fd:
            buf.seek(0)
            shutil.copyfileobj(buf, fd)

        return buf


class Catalogue(Base):
    __tablename__ = 'catalogue'
    __table_args__ = {'autoload': True, 'schema': 'mangasampledb'}

    @property
    def isCurrent(self):
        return self.currentCatalogue is not None

    def __repr__(self):
        return ('<Catalogue (pk={0}), catalogue={1}, version={2}, current={3}>'
                .format(self.pk, self.catalogue_name, self.version,
                        self.isCurrent))


class CurrentCatalogue(Base):
    __tablename__ = 'current_catalogue'
    __table_args__ = {'autoload': True, 'schema': 'mangasampledb'}

    catalogue = relationship(
        'Catalogue', backref=backref('currentCatalogue', uselist=False))

    def __repr__(self):
        return '<CurrentCatalogue (pk={0})>'.format(self.pk)


class MangaTargetToMangaTarget(Base):
    __tablename__ = 'manga_target_to_manga_target'
    __table_args__ = {'autoload': True, 'schema': 'mangasampledb'}

    def __repr__(self):
        return '<MangaTargetToMangaTarget (pk={0})>'.format(self.pk)


class NSA(Base):
    __tablename__ = 'nsa'
    __table_args__ = (
        ForeignKeyConstraint(['catalogue_pk'], ['mangasampledb.catalogue.pk']),
        {'autoload': True, 'schema': 'mangasampledb'})

    def __repr__(self):
        return '<NSA (pk={0}, nsaid={1})>'.format(self.pk, self.nsaid)


class MangaTargetToNSA(Base):
    __tablename__ = 'manga_target_to_nsa'
    __table_args__ = (
        ForeignKeyConstraint(['manga_target_pk'],
                             ['mangasampledb.manga_target.pk']),
        ForeignKeyConstraint(['nsa_pk'], ['mangasampledb.nsa.pk']),
        {'autoload': True, 'schema': 'mangasampledb'})

    def __repr__(self):
        return '<MangaTargetToNSA (pk={0})>'.format(self.pk)


# Relationship between NSA and MangaTarget
NSA.mangaTargets = relationship(
    MangaTarget, backref='NSA_objects', secondary=MangaTargetToNSA.__table__)

# Now we create the remaining tables.
insp = sa_inspect(db.engine)
schemaName = 'mangasampledb'
allTables = insp.get_table_names(schema=schemaName)

done_names = list(db.Base.metadata.tables.keys())
for tableName in allTables:
    if schemaName + '.' + tableName in done_names:
        continue
    className = str(tableName).upper()

    newClass = ClassFactory(
        className, tableName,
        fks=[('catalogue_pk', 'mangasampledb.catalogue.pk')])
    newClass.catalogue = relationship(
        Catalogue, backref='{0}_objects'.format(tableName))
    locals()[className] = newClass
    done_names.append(schemaName + '.' + tableName)

    if 'manga_target_to_' + tableName in allTables:
        relationalTableName = 'manga_target_to_' + tableName
        relationalClassName = 'MangaTargetTo' + tableName.upper()
        newRelationalClass = ClassFactory(
            relationalClassName, relationalTableName,
            fks=[('manga_target_pk', 'mangasampledb.manga_target.pk'),
                 ('nsa_pk', 'mangasampledb.nsa.pk')])

        locals()[relationalClassName] = newRelationalClass
        done_names.append(schemaName + '.' + relationalTableName)

        newClass.mangaTargets = relationship(
            MangaTarget, backref='{0}_objects'.format(tableName),
            secondary=newRelationalClass.__table__)


def HybridMag(flux_parameter, band, index=None):
    """Returns a hybrid property describing an asinh magnitude.

    ``flux_parameter`` must be a column with a flux in nanomaggies. ``band`` is
    the band name, to determine the softening parameter. If ``flux_parameter``
    is and array, ``index`` defines the position of ``band`` within the array.

    """

    @hybrid_property
    def hybridMag(self):
        if index is not None:
            flux = getattr(self, flux_parameter)[index]
        else:
            flux = getattr(self, flux_parameter)

        flux *= 1e-9  # From nanomaggies to maggies
        bb_band = bb[band]
        asinh_mag = -2.5 / np.log(10) * (np.arcsinh(flux / (2. * bb_band)) + np.log(bb_band))
        return asinh_mag

    @hybridMag.expression
    def hybridMag(cls):
        if index is not None:
            # It needs to be index + 1 because Postgresql arrays are 1-indexed.
            flux = getattr(cls, flux_parameter)[index + 1]
        else:
            flux = getattr(cls, flux_parameter)

        flux *= 1e-9
        bb_band = bb[band]
        xx = flux / (2. * bb_band)
        asinh_mag = (-2.5 / func.log(10) *
                     (func.log(xx + func.sqrt(func.pow(xx, 2) + 1)) + func.log(bb_band)))
        return cast(asinh_mag, Float)

    return hybridMag


def HybridColour(parameter):

    @hybrid_method
    def colour(self, bandA, bandB):

        for band in [bandA, bandB]:
            columnName = parameter + '_' + band
            assert hasattr(self, columnName), \
                'cannot find column {0}'.format(columnName)

        bandA_param = getattr(self, parameter + '_' + bandA)
        bandB_param = getattr(self, parameter + '_' + bandB)

        return bandA_param - bandB_param

    @colour.expression
    def colour(cls, bandA, bandB):

        for band in [bandA, bandB]:
            columnName = parameter + '_' + band
            assert hasattr(cls, columnName), \
                'cannot find column {0}'.format(columnName)

        bandA_param = getattr(cls, parameter + '_' + bandA)
        bandB_param = getattr(cls, parameter + '_' + bandB)

        return bandA_param - bandB_param

    return colour


def HybridMethodToProperty(method, bandA, bandB):

    @hybrid_property
    def colour_property(self):
        return getattr(self, method)(bandA, bandB)

    return colour_property


def HybridPropertyArrayIndex(column, index):
    """Creates a hybrid property for a certain index of a column of type array."""

    @hybrid_property
    def array_band(self):
        return getattr(self, column)[index]

    return array_band


# Adds hybrid properties defining asinh magnitudes for each of the SDSS bands.
for ii, band in enumerate('ugriz'):
    propertyName = 'elpetro_mag_{0}'.format(band)
    setattr(NSA, propertyName, HybridMag('elpetro_flux', band, ii + 2))

# Creates attributes for apparent magnitudes.
setattr(NSA, 'elpetro_colour', HybridColour('elpetro_mag'))
for colour_a, colour_b in itertools.combinations('ugriz', 2):
    setattr(NSA, 'elpetro_mag_{0}_{1}'.format(colour_a, colour_b),
            HybridMethodToProperty('elpetro_colour', colour_a, colour_b))

# Expands elpetro_absmag into independent bands.
# TODO: see issue #211 about allowing direct queries on array. If we implement
# that we don't need to expand the bands.
for index, band in enumerate('FNugriz'):
    setattr(NSA, 'elpetro_absmag_{0}'.format(band),
            HybridPropertyArrayIndex('elpetro_absmag', index))

setattr(NSA, 'elpetro_absmag_colour', HybridColour('elpetro_absmag'))
for colour_a, colour_b in itertools.combinations('FNugriz', 2):
    setattr(NSA, 'elpetro_absmag_{0}_{1}'.format(colour_a, colour_b),
            HybridMethodToProperty('elpetro_absmag_colour', colour_a, colour_b))


# Add stellar mass hybrid attributes to NSA catalog
def logmass(parameter):

    @hybrid_property
    def mass(self):
        par = getattr(self, parameter)
        return math.log10(par) if par > 0. else 0.

    @mass.expression
    def mass(cls):
        par = getattr(cls, parameter)
        return cast(case([(par > 0., func.log(par)),
                          (par == 0., 0.)]), Float)

    return mass


setattr(NSA, 'elpetro_logmass', logmass('elpetro_mass'))
setattr(NSA, 'sersic_logmass', logmass('sersic_mass'))
#
configure_mappers()

sample_cache = RelationshipCache(MangaTarget.NSA_objects).\
    and_(RelationshipCache(MangaTarget.character)).\
    and_(RelationshipCache(CurrentCatalogue.catalogue))
