#!/usr/bin/python

# -------------------------------------------------------------------
# Import statements
# -------------------------------------------------------------------
import sys
import os
import math
from decimal import *
from operator import *
from astropy.io import fits

from sqlalchemy.orm import relationship, deferred
from sqlalchemy.schema import Column
from sqlalchemy.engine import reflection
from sqlalchemy.dialects.postgresql import *
from sqlalchemy.types import Float, Integer
from sqlalchemy.orm.session import Session
from sqlalchemy import select, func  # for aggregate, other functions
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy.sql import column
from marvin.db.ArrayUtils import ARRAY_D
from marvin.core.caching_query import RelationshipCache
import numpy as np

try:
    from sdss_access.path import Path
except ImportError as e:
    Path = None

from marvin.db.database import db
import marvin.db.models.SampleModelClasses as sampledb

# ========================
# Define database classes
# ========================
Base = db.Base


class ArrayOps(object):
    ''' this class adds array functionality '''

    __tablename__ = 'arrayops'
    __table_args__ = {'extend_existing': True}

    @property
    def cols(self):
        return list(self.__table__.columns._data.keys())

    @property
    def collist(self):
        return ['wavelength', 'flux', 'ivar', 'mask', 'xpos', 'ypos', 'specres']

    def getTableName(self):
        return self.__table__.name

    def matchIndex(self, name=None):

        # Get index of correct column
        incols = [x for x in self.cols if x in self.collist]
        if not any(incols):
            return None
        elif len(incols) == 1:
            idx = self.cols.index(incols[0])
        else:
            if not name:
                print('Multiple columns found.  Column name must be specified!')
                return None
            elif name in self.collist:
                idx = self.cols.index(name)
            else:
                return None

        return idx

    def filter(self, start, end, name=None):

        # Check input types or map string operators
        startnum = type(start) == int or type(start) == float
        endnum = type(end) == int or type(end) == float
        opdict = {'=': eq, '<': lt, '<=': le, '>': gt, '>=': ge, '!=': ne}
        if start in opdict.keys() or end in opdict.keys():
            opind = list(opdict.keys()).index(start) if start in opdict.keys() else list(opdict.keys()).index(end)
            if start in opdict.keys():
                start = opdict[list(opdict.keys())[opind]]
            if end in opdict.keys():
                end = opdict[list(opdict.keys())[opind]]

        # Get matching index
        self.idx = self.matchIndex(name=name)
        if not self.idx:
            return None

        # Perform calculation
        try:
            data = self.__getattribute__(self.cols[self.idx])
        except:
            data = None

        if data:
            if startnum and endnum:
                arr = [x for x in data if x >= start and x <= end]
            elif not startnum and endnum:
                arr = [x for x in data if start(x, end)]
            elif startnum and not endnum:
                arr = [x for x in data if end(x, start)]
            elif startnum == eq or endnum == eq:
                arr = [x for x in data if start(x, end)] if start == eq else [x for x in data if end(x, start)]
            return arr
        else:
            return None

    def equal(self, num, name=None):

        # Get matching index
        self.idx = self.matchIndex(name=name)
        if not self.idx:
            return None

        # Perform calculation
        try:
            data = self.__getattribute__(self.cols[self.idx])
        except:
            data = None

        if data:
            arr = [x for x in data if x == num]
            return arr
        else:
            return None

    def less(self, num, name=None):

        # Get matching index
        self.idx = self.matchIndex(name=name)
        if not self.idx:
            return None

        # Perform calculation
        try:
            data = self.__getattribute__(self.cols[self.idx])
        except:
            data = None

        if data:
            arr = [x for x in data if x <= num]
            return arr
        else:
            return None

    def greater(self, num, name=None):

        # Get matching index
        self.idx = self.matchIndex(name=name)
        if not self.idx:
            return None

        # Perform calculation
        try:
            data = self.__getattribute__(self.cols[self.idx])
        except:
            data = None

        if data:
            arr = [x for x in data if x >= num]
            return arr
        else:
            return None

    def getIndices(self, arr):

        if self.idx:
            indices = [self.__getattribute__(self.cols[self.idx]).index(a) for a in arr]
        else:
            return None

        return indices


class Cube(Base, ArrayOps):
    __tablename__ = 'cube'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb', 'extend_existing': True}

    specres = deferred(Column(ARRAY_D(Float, zero_indexes=True)))

    def __repr__(self):
        return '<Cube (pk={0}, plate={1}, ifudesign={2}, tag={3})>'.format(self.pk, self.plate, self.ifu.name, self.pipelineInfo.version.version)

    @property
    def header(self):
        '''Returns an astropy header'''

        session = Session.object_session(self)
        data = session.query(FitsHeaderKeyword.label, FitsHeaderValue.value,
                             FitsHeaderValue.comment).join(FitsHeaderValue).filter(
            FitsHeaderValue.cube == self).all()

        hdr = fits.Header(data)
        return hdr

    @property
    def name(self):
        return 'manga-{0}-{1}-LOGCUBE.fits.gz'.format(self.plate, self.ifu.name)

    @property
    def default_mapsname(self):
        return 'mangadap-{0}-{1}-default.fits.gz'.format(self.plate, self.ifu.name)

    def getPath(self):
        sasurl = os.getenv('SAS_URL')

        if sasurl:
            sasredux = os.path.join(sasurl, 'sas/mangawork/manga/spectro/redux')
            path = sasredux
        else:
            redux = os.getenv('MANGA_SPECTRO_REDUX')
            path = redux

        version = self.pipelineInfo.version.version
        cubepath = os.path.join(path, version, str(self.plate), 'stack')
        return cubepath

    @property
    def location(self):
        name = self.name
        path = self.getPath()
        loc = os.path.join(path, name)
        return loc

    @property
    def image(self):
        ifu = '{0}.png'.format(self.ifu.name)
        path = self.getPath()
        imageloc = os.path.join(path, 'images', ifu)
        return imageloc

    def header_to_dict(self):
        '''Returns a simple python dictionary header'''

        values = self.headervals
        hdrdict = {str(val.keyword.label): val.value for val in values}
        return hdrdict

    @property
    def plateclass(self):
        '''Returns a plate class'''

        plate = Plate(self)

        return plate

    def testhead(self, key):
        ''' Test existence of header keyword'''

        try:
            if self.header_to_dict()[key]:
                return True
        except:
            return False

    def getFlags(self, bits, name):
        session = Session.object_session(self)

        # if bits not a digit, return None
        if not str(bits).isdigit():
            return 'NULL'
        else:
            bits = int(bits)

        # Convert the integer value to list of bits
        bitlist = [int(i) for i in '{0:08b}'.format(bits)]
        bitlist.reverse()
        indices = [i for i, bit in enumerate(bitlist) if bit]

        labels = []
        for i in indices:
            maskbit = session.query(MaskBit).filter_by(flag=name, bit=i).one()
            labels.append(maskbit.label)

        return labels

    def getQualFlags(self, stage='3d'):
        ''' get quality flags '''

        name = 'MANGA_DRP2QUAL' if stage == '2d' else 'MANGA_DRP3QUAL'
        col = 'DRP2QUAL' if stage == '2d' else 'DRP3QUAL'
        try:
            bits = self.header_to_dict()[col]
        except:
            bits = None

        if bits:
            labels = self.getFlags(bits, name)
            return labels
        else:
            return None

    def getTargFlags(self, type=1):
        ''' get target flags '''

        name = 'MANGA_TARGET1' if type == 1 else 'MANGA_TARGET2' if type == 2 else 'MANGA_TARGET3'
        hdr = self.header_to_dict()
        istarg = 'MNGTARG1' in hdr.keys()
        if istarg:
            col = 'MNGTARG1' if type == 1 else 'MNGTARG2' if type == 2 else 'MNGTARG3'
        else:
            col = 'MNGTRG1' if type == 1 else 'MNGTRG2' if type == 2 else 'MNGTRG3'

        try:
            bits = hdr[col]
        except:
            bits = None

        if bits:
            labels = self.getFlags(bits, name)
            return labels
        else:
            return None

    def get3DCube(self, extension='flux'):
        """Returns a 3D array of ``extension`` from the cube spaxels.

        For example, ``cube.get3DCube('flux')`` will return the original
        flux cube with the same ordering as the FITS data cube.

        Note that this method seems to be really slow retrieving arrays (this
        is especially serious for large IFUs).

        """

        session = Session.object_session(self)
        spaxels = session.query(getattr(Spaxel, extension)).filter(
            Spaxel.cube_pk == self.pk).order_by(Spaxel.x, Spaxel.y).all()

        # Assumes cubes are always square (!)
        nx = ny = int(np.sqrt(len(spaxels)))
        nwave = len(spaxels[0][0])

        spArray = np.array(spaxels)

        return spArray.transpose().reshape((nwave, ny, nx)).transpose(0, 2, 1)

    @hybrid_property
    def plateifu(self):
        '''Returns parameter plate-ifu'''
        return '{0}-{1}'.format(self.plate, self.ifu.name)

    @plateifu.expression
    def plateifu(cls):
        return func.concat(Cube.plate, '-', IFUDesign.name)

    @hybrid_property
    def restwave(self):
        if self.target:
            redshift = self.target.NSA_objects[0].z
            wave = np.array(self.wavelength.wavelength)
            restwave = wave/(1+redshift)
            return restwave
        else:
            return None

    @restwave.expression
    def restwave(cls):
        restw = (func.rest_wavelength(sampledb.NSA.z))
        return restw


class Wavelength(Base, ArrayOps):
    __tablename__ = 'wavelength'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb', 'extend_existing': True}

    wavelength = deferred(Column(ARRAY_D(Float, zero_indexes=True)))

    def __repr__(self):
        return '<Wavelength (pk={0})>'.format(self.pk)


class Spaxel(Base, ArrayOps):
    __tablename__ = 'spaxel'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb', 'extend_existing': True}

    flux = deferred(Column(ARRAY_D(Float, zero_indexes=True)))
    ivar = deferred(Column(ARRAY_D(Float, zero_indexes=True)))
    mask = deferred(Column(ARRAY_D(Integer, zero_indexes=True)))

    def __repr__(self):
        return '<Spaxel (pk={0}, x={1}, y={2})'.format(self.pk, self.x, self.y)

    @hybrid_method
    def sum(self, name=None):

        total = sum(self.flux)

        return total

    @sum.expression
    def sum(cls):
        # return select(func.sum(func.unnest(cls.flux))).select_from(func.unnest(cls.flux)).label('totalflux')
        return select([func.sum(column('totalflux'))]).select_from(func.unnest(cls.flux).alias('totalflux'))


class RssFiber(Base, ArrayOps):
    __tablename__ = 'rssfiber'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb', 'extend_existing': True}

    flux = deferred(Column(ARRAY_D(Float, zero_indexes=True)))
    ivar = deferred(Column(ARRAY_D(Float, zero_indexes=True)))
    mask = deferred(Column(ARRAY_D(Integer, zero_indexes=True)))
    xpos = deferred(Column(ARRAY_D(Float, zero_indexes=True)))
    ypos = deferred(Column(ARRAY_D(Float, zero_indexes=True)))

    def __repr__(self):
        return '<RssFiber (pk={0})>'.format(self.pk)


class PipelineInfo(Base):
    __tablename__ = 'pipeline_info'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Pipeline_Info (pk={0})>'.format(self.pk)


class PipelineVersion(Base):
    __tablename__ = 'pipeline_version'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Pipeline_Version (pk={0}, version={1})>'.format(self.pk, self.version)


class PipelineStage(Base):
    __tablename__ = 'pipeline_stage'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Pipeline_Stage (pk={0}, label={1})>'.format(self.pk, self.label)


class PipelineCompletionStatus(Base):
    __tablename__ = 'pipeline_completion_status'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Pipeline_Completion_Status (pk={0}, label={1})>'.format(self.pk, self.label)


class PipelineName(Base):
    __tablename__ = 'pipeline_name'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Pipeline_Name (pk={0}, label={1})>'.format(self.pk, self.label)


class FitsHeaderValue(Base):
    __tablename__ = 'fits_header_value'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Fits_Header_Value (pk={0})'.format(self.pk)


class FitsHeaderKeyword(Base):
    __tablename__ = 'fits_header_keyword'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Fits_Header_Keyword (pk={0}, label={1})'.format(self.pk, self.label)


class IFUDesign(Base):
    __tablename__ = 'ifudesign'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<IFU_Design (pk={0}, name={1})'.format(self.pk, self.name)

    @property
    def ifuname(self):
        return self.name

    @property
    def designid(self):
        return self.name

    @property
    def ifutype(self):
        return self.name[:-2]


class IFUToBlock(Base):
    __tablename__ = 'ifu_to_block'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<IFU_to_Block (pk={0})'.format(self.pk)


class SlitBlock(Base):
    __tablename__ = 'slitblock'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<SlitBlock (pk={0})'.format(self.pk)


class Cart(Base):
    __tablename__ = 'cart'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Cart (pk={0}, id={1})'.format(self.pk, self.id)


class Fibers(Base):
    __tablename__ = 'fibers'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Fibers (pk={0}, fiberid={1}, fnum={2})'.format(self.pk, self.fiberid, self.fnum)


class FiberType(Base):
    __tablename__ = 'fiber_type'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Fiber_Type (pk={0},label={1})'.format(self.pk, self.label)


class TargetType(Base):
    __tablename__ = 'target_type'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Target_Type (pk={0},label={1})'.format(self.pk, self.label)


class Sample(Base, ArrayOps):
    __tablename__ = 'sample'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<Sample (pk={0},cube={1})'.format(self.pk, self.cube)

    @hybrid_property
    def nsa_logmstar(self):
        try:
            return math.log10(self.nsa_mstar)
        except ValueError:
            return -9999.0
        except TypeError:
            return None

    @nsa_logmstar.expression
    def nsa_logmstar(cls):
        return func.log(cls.nsa_mstar)

    @hybrid_property
    def nsa_logmstar_el(self):
        try:
            return math.log10(self.nsa_mstar_el)
        except ValueError as e:
            return -9999.0
        except TypeError as e:
            return None

    @nsa_logmstar_el.expression
    def nsa_logmstar_el(cls):
        return func.log(cls.nsa_mstar_el)


class CartToCube(Base):
    __tablename__ = 'cart_to_cube'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<CartToCube (pk={0},cube={1}, cart={2})'.format(self.pk, self.cube, self.cart)


class Wcs(Base, ArrayOps):
    __tablename__ = 'wcs'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<WCS (pk={0},cube={1})'.format(self.pk, self.cube)

    def makeHeader(self):
        wcscols = self.cols[2:]
        newhdr = fits.Header()
        for c in wcscols:
            newhdr[c] = float(self.__getattribute__(c)) if type(self.__getattribute__(c)) == Decimal else self.__getattribute__(c)
        return newhdr


class CubeShape(Base):
    __tablename__ = 'cube_shape'
    __table_args__ = {'autoload': True, 'schema': 'mangadatadb'}

    def __repr__(self):
        return '<CubeShape (pk={0},cubes={1},size={2},totalrows={3})'.format(self.pk, len(self.cubes), self.size, self.total)

    @property
    def shape(self):
        return (self.size, self.size)

    def makeIndiceArray(self):
        ''' Return the indices array as a numpy array '''
        return np.array(self.indices)

    def getXY(self, index=None):
        ''' Get the x,y elements from a single digit index '''
        if index is not None:
            if index > self.total:
                return None, None
            else:
                i = int(index / self.size)
                j = int(index - i * self.size)
        else:
            arrind = self.makeIndiceArray()
            i = np.array(arrind / self.size, dtype=int)
            j = np.array(self.makeIndiceArray() - i * self.size, dtype=int)
        return i, j

    @hybrid_property
    def x(self):
        '''Returns parameter plate-ifu'''
        x = self.getXY()[0]
        return x

    @x.expression
    def x(cls):
        #arrind = func.unnest(cls.indices).label('arrind')
        #x = func.array_agg(arrind / cls.size).label('x')
        s = db.Session()
        arrind = (func.unnest(cls.indices) / cls.size).label('xarrind')
        #x = s.query(arrind).select_from(cls).subquery('xarr')
        #xagg = s.query(func.array_agg(x.c.xarrind))
        return arrind

    @hybrid_property
    def y(self):
        '''Returns parameter plate-ifu'''
        y = self.getXY()[1]
        return y

    @y.expression
    def y(cls):
        #arrind = func.unnest(cls.indices).label('arrind')
        #x = arrind / cls.size
        #y = func.array_agg(arrind - x*cls.size).label('y')
        #return y
        s = db.Session()
        arrunnest = func.unnest(cls.indices)
        xarr = (func.unnest(cls.indices) / cls.size).label('xarrind')
        arrind = (arrunnest - xarr*cls.size).label('yarrind')
        #n.arrind-(n.arrind/n.size)*n.size
        y = s.query(arrind).select_from(cls).subquery('yarr')
        yagg = s.query(func.array_agg(y.c.yarrind))
        return yagg.as_scalar()


class Plate(object):
    ''' new plate class '''

    __tablename__ = 'myplate'

    def __init__(self, cube=None, id=None):
        self.id = cube.plate if cube else id if id else None
        self.cube = cube if cube else None
        self.drpver = None
        if self.cube:
            self._hdr = self.cube.header_to_dict()
            self.type = self.getPlateType()
            self.platetype = self._hdr.get('PLATETYP', None)
            self.surveymode = self._hdr.get('SRVYMODE', None)
            self.dateobs = self._hdr.get('DATE-OBS', None)
            self.ra = self._hdr.get('CENRA', None)
            self.dec = self._hdr.get('CENDEC', None)
            self.designid = self._hdr.get('DESIGNID', None)
            self.cartid = self._hdr.get('CARTID', None)
            self.drpver = self.cube.pipelineInfo.version.version
            self.isbright = 'APOGEE' in self.surveymode
            self.dir3d = 'mastar' if self.isbright else 'stack'

            # cast a few
            self.ra = float(self.ra) if self.ra else None
            self.dec = float(self.dec) if self.dec else None
            self.id = int(self.id) if self.id else None
            self.designid = int(self.designid) if self.designid else None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ('Plate (id={0}, ra={1}, dec={2}, '
                ' designid={3})'.format(self.id, self.ra, self.dec, self.designid))

    def getPlateType(self):
        ''' Get the type of MaNGA plate '''

        hdr = self.cube.header

        # try galaxy
        mngtrg = self._hdr.get('MNGTRG1', None)
        pltype = 'Galaxy' if mngtrg else None

        # try stellar
        if not pltype:
            mngtrg = self._hdr.get('MNGTRG2', None)
            pltype = 'Stellar' if mngtrg else None

        # try ancillary
        if not pltype:
            mngtrg = self._hdr.get('MNGTRG3', None)
            pltype = 'Ancillary' if mngtrg else None

        return pltype

    @property
    def cubes(self):
        ''' Get all cubes on this plate '''

        session = db.Session()
        if self.drpver:
            cubes = session.query(Cube).join(PipelineInfo, PipelineVersion).\
                filter(Cube.plate == self.id, PipelineVersion.version == self.drpver).all()
        else:
            cubes = session.query(Cube).filter(Cube.plate == self.id).all()
        return cubes


# ================
# manga Aux DB classes
# ================
class CubeHeader(Base):
    __tablename__ = 'cube_header'
    __table_args__ = {'autoload': True, 'schema': 'mangaauxdb'}

    def __repr__(self):
        return '<CubeHeader (pk={0},cube={1})'.format(self.pk, self.cube)


class MaskLabels(Base):
    __tablename__ = 'maskbit_labels'
    __table_args__ = {'autoload': True, 'schema': 'mangaauxdb'}

    def __repr__(self):
        return '<MaskLabels (pk={0},bit={1})'.format(self.pk, self.maskbit)


class MaskBit(Base):
    __tablename__ = 'maskbit'
    __table_args__ = {'autoload': True, 'schema': 'mangaauxdb'}

    def __repr__(self):
        return '<MaskBit (pk={0},flag={1}, bit={2}, label={3})'.format(self.pk, self.flag, self.bit, self.label)

# ========================
# Define relationships
# ========================

Cube.pipelineInfo = relationship(PipelineInfo, backref="cubes")
Cube.wavelength = relationship(Wavelength, backref="cube")
Cube.ifu = relationship(IFUDesign, backref="cubes")
Cube.carts = relationship(Cart, secondary=CartToCube.__table__, backref="cubes")
Cube.wcs = relationship(Wcs, backref='cube', uselist=False)
Cube.shape = relationship(CubeShape, backref='cubes', uselist=False)
# from SampleDB
Cube.target = relationship(sampledb.MangaTarget, backref='cubes')

Sample.cube = relationship(Cube, backref="sample", uselist=False)

FitsHeaderValue.cube = relationship(Cube, backref="headervals")
FitsHeaderValue.keyword = relationship(FitsHeaderKeyword, backref="value")

IFUDesign.blocks = relationship(SlitBlock, secondary=IFUToBlock.__table__, backref='ifus')
Fibers.ifu = relationship(IFUDesign, backref="fibers")
Fibers.fibertype = relationship(FiberType, backref="fibers")
Fibers.targettype = relationship(TargetType, backref="fibers")

insp = reflection.Inspector.from_engine(db.engine)
fks = insp.get_foreign_keys(Spaxel.__table__.name, schema='mangadatadb')
if fks:
    Spaxel.cube = relationship(Cube, backref='spaxels')
fks = insp.get_foreign_keys(RssFiber.__table__.name, schema='mangadatadb')
if fks:
    RssFiber.cube = relationship(Cube, backref='rssfibers')
    RssFiber.fiber = relationship(Fibers, backref='rssfibers')

PipelineInfo.name = relationship(PipelineName, backref="pipeinfo")
PipelineInfo.stage = relationship(PipelineStage, backref="pipeinfo")
PipelineInfo.version = relationship(PipelineVersion, backref="pipeinfo")
PipelineInfo.completionStatus = relationship(PipelineCompletionStatus, backref="pipeinfo")

# from AuxDB
CubeHeader.cube = relationship(Cube, backref='hdr')

# ---------------------------------------------------------
# Test that all relationships/mappings are self-consistent.
# ---------------------------------------------------------
from sqlalchemy.orm import configure_mappers
try:
    configure_mappers()
except RuntimeError as error:
    print("""
mangadb.DataModelClasses:
An error occurred when verifying the relationships between the database tables.
Most likely this is an error in the definition of the SQLAlchemy relationships -
see the error message below for details.
""")
    print("Error type: %s" % sys.exc_info()[0])
    print("Error value: %s" % sys.exc_info()[1])
    print("Error trace: %s" % sys.exc_info()[2])
    sys.exit(1)


data_cache = RelationshipCache(Cube.target).\
               and_(RelationshipCache(Cube.ifu)).\
               and_(RelationshipCache(Cube.spaxels)).\
               and_(RelationshipCache(Cube.wavelength)).\
               and_(RelationshipCache(IFUDesign.fibers)).\
               and_(RelationshipCache(Cube.rssfibers))
