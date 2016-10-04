#!/usr/bin/env python
# encoding: utf-8

'''
Created by Brian Cherinka on 2016-05-23 00:13:58
Licensed under a 3-clause BSD license.

Revision History:
    Initial Version: 2016-05-23 00:13:58 by Brian Cherinka
    Last Modified On: 2016-05-23 00:13:58 by Brian

'''
from __future__ import print_function
from __future__ import division
from sqlalchemy import type_coerce
from sqlalchemy.dialects.postgresql import *


# __getitem__ broken for postgres ARRAY type
#
# this is a bug that's been fixed for 1.1.   It's detailed here:
# http://docs.sqlalchemy.org/en/latest/changelog/migration_11.html#correct-sql-types-are-established-from-indexed-access-of-array-json-hstore

# For multi-dimensional access, this can be worked around for a one-off
# using type_coerce:

#  >>> from sqlalchemy import type_coerce
#  >>> type_coerce(c[4], ARRAY(Integer))[5]

# There is also a generalized workaround created for the bug that you can
# see at
# https://bitbucket.org/zzzeek/sqlalchemy/issues/3487#comment-20200804 .
# It involves creation of an ARRAY subclass that does the right thing
# within __getitem__.   That subclass can be a drop-in replacement for
# regular ARRAY.


class ARRAY_D(ARRAY):
    class Comparator(ARRAY.Comparator):
        def __getitem__(self, index):
            super_ = super(ARRAY_D.Comparator, self).__getitem__(index)
            if not isinstance(index, slice) and self.type.dimensions > 1:
                super_ = type_coerce(
                    super_,
                    ARRAY_D(
                        self.type.item_type,
                        dimensions=self.type.dimensions - 1,
                        zero_indexes=self.type.zero_indexes)
                )
            return super_
    comparator_factory = Comparator
