#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-04-26 09:20:35
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-04-26 09:20:35 by Brian Cherinka
    Last Modified On: 2016-04-26 09:20:35 by Brian
'''

import numpy
from decimal import Decimal
from psycopg2.extensions import register_adapter, AsIs, new_type, DECIMAL, register_type

# See:
# http://rehalcon.blogspot.com/2010/03/sqlalchemy-programmingerror-cant-adapt.html
# and
# http://initd.org/psycopg/docs/advanced.html#adapting-new-python-types-to-sql-syntax
# and
# http://pyopengl.sourceforge.net/pydoc/numpy.core.numerictypes.html
#
# http://numpy.sourceforge.net/numdoc/HTML/numdoc.htm

''' numpy data types:
int8 int16 int32 int64 int128
uint8 uint16 uint32 uint64 uint128
float16 float32 float64 float96 float128 float256
complex32 complex64 complex128 complex192 complex256 complex512
'''

DEC2FLOAT = new_type(DECIMAL.values, 'DEC2FLOAT', lambda value,
                     curs: float(value) if value is not None else None)
register_type(DEC2FLOAT)


def adapt_decimal(Decimal):
    return AsIs(float)
register_adapter(Decimal, adapt_decimal)


def adapt_numpy_int8(numpy_int8):
    return AsIs(numpy_int8)
register_adapter(numpy.int8, adapt_numpy_int8)


def adapt_numpy_int16(numpy_int16):
    return AsIs(numpy_int16)
register_adapter(numpy.int16, adapt_numpy_int16)


def adapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)
register_adapter(numpy.int32, adapt_numpy_int32)


def adapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(numpy.int64, adapt_numpy_int64)


# def adapt_numpy_int128(numpy_int128):
#   return AsIs(numpy_int128)
# register_adapter(numpy.int128, adapt_numpy_int128)


def adapt_numpy_uint8(numpy_uint8):
    return AsIs(numpy_uint8)
register_adapter(numpy.uint8, adapt_numpy_uint8)


def adapt_numpy_uint16(numpy_uint16):
    return AsIs(numpy_uint16)
register_adapter(numpy.uint16, adapt_numpy_uint16)


def adapt_numpy_uint32(numpy_uint32):
    return AsIs(numpy_uint32)
register_adapter(numpy.uint32, adapt_numpy_uint32)


def adapt_numpy_uint64(numpy_uint64):
    return AsIs(numpy_uint64)
register_adapter(numpy.uint64, adapt_numpy_uint64)


# def adapt_numpy_uint128(numpy_uint128):
#   return AsIs(numpy_uint128)
# register_adapter(numpy.uint128, adapt_numpy_uint128)


# def adapt_numpy_float16(numpy_float16):
#   return AsIs(numpy_float16)
# register_adapter(numpy.float16, adapt_numpy_float16)

def adapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)
register_adapter(numpy.float32, adapt_numpy_float32)


def adapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
register_adapter(numpy.float64, adapt_numpy_float64)


# def adapt_numpy_float96(numpy_float96):
#   return AsIs(numpy_float96)
# register_adapter(numpy.float96, adapt_numpy_float96)


# def adapt_numpy_float128(numpy_float128):
#   return AsIs(numpy_float128)
# register_adapter(numpy.float128, adapt_numpy_float128)


# def adapt_numpy_float256(numpy_float256):
#   return AsIs(numpy_float256)
# register_adapter(numpy.float256, adapt_numpy_float256)


# def adapt_numpy_complex32(numpy_complex32):
#   return AsIs(numpy_complex32)
# register_adapter(numpy.complex32, adapt_numpy_complex32)


# def adapt_numpy_complex64(numpy_complex64):
#   return AsIs(numpy_complex64)
# register_adapter(numpy.complex64, adapt_numpy_complex64)


# def adapt_numpy_complex128(numpy_complex128):
#   return AsIs(numpy_complex128)
# register_adapter(numpy.complex128, adapt_numpy_complex128)


# def adapt_numpy_complex192(numpy_complex192):
#   return AsIs(numpy_complex192)
# register_adapter(numpy.complex192, adapt_numpy_complex192)


# def adapt_numpy_complex256(numpy_complex256):
#   return AsIs(numpy_complex256)
# register_adapter(numpy.complex256, adapt_numpy_complex256)


# def adapt_numpy_complex512(numpy_complex512):
#   return AsIs(numpy_complex512)
# register_adapter(numpy.complex512, adapt_numpy_complex512)

def adapt_numpy_bool(numpy_bool):
    return AsIs(numpy_bool)
register_adapter(numpy.bool_, adapt_numpy_bool)


def adapt_numpy_nan(numpy_nan):
    return "'NaN'"
register_adapter(numpy.nan, adapt_numpy_nan)


def adapt_numpy_inf(numpy_inf):
    return "'Infinity'"
register_adapter(numpy.inf, adapt_numpy_inf)


def adapt_numpy_ndarray(numpy_ndarray):
    return AsIs(numpy_ndarray.tolist())
register_adapter(numpy.ndarray, adapt_numpy_ndarray)
